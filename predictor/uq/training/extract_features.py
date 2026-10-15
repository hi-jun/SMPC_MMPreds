"""Extract STDAN encoder features for conformal novelty detection.

Runs the frozen trunk over an NGSIM split and stores, per sample, four
candidate feature taps plus the intent softmax and logits:

    agent        (64)  LSTM state of the target's own history        (StreamTap)
    social       (64)  spatial-attention GLU output                   (StreamTap)
    values_last  (64)  encoder output at the last step (what Generator consumes)
    pre_logit    (64)  maneuver_state = input of op_lat, the intent head's pre-logit
    lat_probs    (3)   STDAN lateral softmax          logits (3)  op_lat output

Sampling is the whole point of this file. Adjacent frames of one vehicle are
near-identical, so drawing frames at random makes calibration points' nearest
neighbours their own temporal neighbours, kNN distances collapse toward zero
and deployed p-values come out systematically small (a reviewer's warning we
took seriously). The sampler therefore:

  * keeps whole TRACKS together - the NGSIM splits are already track-disjoint
    (verified: Train/Val share 6 of 944 tracks, Val/Test 6 of 2022; those are
    dropped here), so reference / calibration / test never share a vehicle;
  * thins WITHIN a track to one sample per `--gap` seconds (default 3 s), so
    a track of median 80 s contributes ~27 points instead of ~800 duplicates;
  * draws WITHOUT replacement and WITHOUT class balancing - the calibration
    set must be exchangeable with what the detector will see, and the
    existing balanced_loader (replacement=True) is neither.

Domain-shift batches (--shift SCALE) reuse compare_heads.py's probe: multiply
the position/velocity channels (hist, nbrs, va, nbrsva) by SCALE.

Run (hmg env, from this directory):
    python extract_features.py --split Val --gap 3 --out ../ckpt/features/cal_val.npz
    python extract_features.py --split Test --gap 3 --shift 2.5 --out .../test_shift_2.5.npz
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
# config_vel / loader2_hdf5_vel / model5f_mult_vel: this repository's
# predictor/stdan_vel copies (same model and loader as the source repository)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "stdan_vel")))

from config_vel import args                        # noqa: E402
import loader2_hdf5_vel as lo                      # noqa: E402
from model5f_mult_vel import GDEncoder, Generator   # noqa: E402
from predictor.uq.stream_tap import StreamTap, encoder_inputs  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
# the trunk this repository runs as stdan_vel (= ensemble member0)
STDAN_VEL_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "stdan_vel", "checkpoints", "0113_ratio211", "best_model.pt")
FT2M = 0.3048
SHIFT_CHANNELS = (0, 1, 3, 4)   # hist, nbrs, va, nbrsva  (compare_heads.py:61)


def load_trunk(device, ckpt=None):
    args.setdefault("train_flag", False)
    gd = GDEncoder(args).to(device)
    ckpt = ckpt or STDAN_VEL_CKPT
    if os.path.isdir(ckpt):
        gd_f = sorted(glob.glob(os.path.join(ckpt, "*_gd.tar")))[-1]
        g_f = sorted(glob.glob(os.path.join(ckpt, "*_g.tar")))[-1]
        gd_state = torch.load(gd_f, map_location=device, weights_only=False)
        g_state = torch.load(g_f, map_location=device, weights_only=False)
    else:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        gd_state, g_state = ck["gdEncoder"], ck["generator"]
    args["out_length"] = int(g_state["mapping"].shape[1])
    gen = Generator(args).to(device)
    gd.load_state_dict(gd_state)
    gen.load_state_dict(g_state)
    gd.eval(); gen.eval()
    return gd, gen


# ------------------------------------------------------------- track sampler


def track_ids(ds):
    """(dsId, vehId) per row of the split, as one int64 key."""
    D = ds.D
    return (D[:, 0].astype(np.int64) * 100000 + D[:, 1].astype(np.int64))


def thinned_indices(ds, gap_s, exclude_tracks=(), max_samples=None,
                    one_per_track=False, seed=0):
    """Row indices: every track, thinned to >= gap_s seconds between picks.

    Rows are 10 Hz. Picks start at a random phase per track so that thinning
    does not always land on the same frame numbers. `max_samples` keeps every
    track represented (round-robin over tracks) rather than truncating.
    """
    D = ds.D
    keys = track_ids(ds)
    frames = D[:, 2].astype(np.int64)
    rng = np.random.RandomState(seed)
    order = np.lexsort((frames, keys))
    keys_s, frames_s = keys[order], frames[order]
    bounds = np.flatnonzero(np.diff(keys_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(keys_s)]])
    step = max(1, int(round(gap_s * 10)))
    excl = set(int(k) for k in exclude_tracks)
    per_track = []
    for a, b in zip(starts, ends):
        if int(keys_s[a]) in excl:
            continue
        rows = order[a:b]
        if one_per_track:
            per_track.append(rows[rng.randint(len(rows))][None])
            continue
        phase = rng.randint(step) if len(rows) > step else 0
        per_track.append(rows[phase::step])
    if max_samples is not None and sum(len(r) for r in per_track) > max_samples:
        # round-robin so that every track keeps at least its first pick
        out, k = [], 0
        while len(out) < max_samples:
            took = False
            for r in per_track:
                if k < len(r):
                    out.append(r[k]); took = True
                    if len(out) >= max_samples:
                        break
            if not took:
                break
            k += 1
        return np.asarray(sorted(out), dtype=np.int64)
    return np.asarray(sorted(np.concatenate(per_track)), dtype=np.int64)


def overlapping_tracks(mat_a, mat_b):
    import scipy.io as sio
    def keys(m):
        try:
            D = np.asarray(sio.loadmat(m)["traj"])
        except Exception:
            import h5py
            D = np.asarray(h5py.File(m, "r")["traj"]).T
        return set((D[:, 0].astype(np.int64) * 100000 + D[:, 1].astype(np.int64)).tolist())
    return keys(mat_a) & keys(mat_b)


# ---------------------------------------------------------------- extraction


class PreLogitTap:
    """Grab the input of generator.op_lat (= maneuver_state) and its output."""

    def __init__(self, gen):
        self._x = None; self._y = None
        self._h = [gen.op_lat.register_forward_pre_hook(self._pre),
                   gen.op_lat.register_forward_hook(self._post)]

    def _pre(self, module, inputs):
        self._x = inputs[0].detach()

    def _post(self, module, inputs, output):
        self._y = output.detach()

    def take(self):
        x, y = self._x, self._y
        self._x = self._y = None
        return x, y

    def remove(self):
        for h in self._h:
            h.remove()


def extract(ds, idx, gd, gen, device, batch=128, shift=1.0, workers=2):
    tap = StreamTap(gd)
    plt = PreLogitTap(gen)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=ds.collate_fn)
    outs = {k: [] for k in ("agent", "social", "values_last", "pre_logit",
                            "logits", "lat_probs", "labels", "vlat")}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch_ in enumerate(loader):
            hist = batch_[0]
            enc_in, lat_enc = encoder_inputs(batch_, device)
            if shift != 1.0:
                enc_in = list(enc_in)
                for c in SHIFT_CHANNELS:
                    enc_in[c] = enc_in[c] * shift
                enc_in = tuple(enc_in)
            values = gd(*enc_in)
            if isinstance(values, tuple):
                values = values[0]
            agent, social = tap.features()
            z_lat = torch.zeros(values.shape[0], args["lat_length"], device=device)
            z_lon = torch.zeros(values.shape[0], args["lon_length"], device=device)
            out = gen(values, z_lat, z_lon)
            lat_pred = out[1]
            pre, logit = plt.take()
            outs["agent"].append(agent.cpu().numpy())
            outs["social"].append(social.cpu().numpy())
            outs["values_last"].append(values[:, -1, :].cpu().numpy())
            outs["pre_logit"].append(pre.cpu().numpy())
            outs["logits"].append(logit.cpu().numpy())
            outs["lat_probs"].append(lat_pred.detach().cpu().numpy())
            outs["labels"].append(lat_enc.argmax(-1).cpu().numpy())
            h = hist.numpy()                              # (T, B, 2), col 0 lateral (ft)
            outs["vlat"].append((h[-1, :, 0] - h[-6, :, 0]) * FT2M / 0.5)
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)}  {time.time()-t0:.0f}s", flush=True)
    tap.remove(); plt.remove()
    return {k: np.concatenate(v, 0) for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/core-dev/HJ/dataset/raw_data_ngsim/dataset_t_v_t")
    ap.add_argument("--split", default="Val", choices=["Train", "Val", "Test"])
    ap.add_argument("--highd", nargs="*", type=int, default=None,
                    help="recording ids: build samples from the RAW highD CSVs "
                         "instead of an NGSIM split (the shipped highD .mat "
                         "preprocessing has an unusable social grid)")
    ap.add_argument("--highd-stride", type=float, default=2.0)
    ap.add_argument("--highd-per-rec", type=int, default=1500)
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between picks in a track")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--one-per-track", action="store_true")
    ap.add_argument("--exclude-overlap-with", nargs="*", default=[],
                    help="other splits whose tracks must be dropped from this one")
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="trunk checkpoint: pair dir or best_model.pt "
                         "(default: predictor/stdan_vel/checkpoints/0113_ratio211/best_model.pt)")
    cli = ap.parse_args()

    device = args["device"]
    torch.manual_seed(72)
    gd, gen = load_trunk(device, ckpt=cli.ckpt)

    if cli.highd:
        from highd_raw import HighDRaw
        ds = HighDRaw(cli.highd, stride_s=cli.highd_stride,
                      max_per_recording=cli.highd_per_rec, seed=cli.seed)
        idx = np.arange(len(ds), dtype=np.int64)
        if cli.max_samples and len(idx) > cli.max_samples:
            idx = np.sort(np.random.RandomState(cli.seed).choice(
                len(idx), cli.max_samples, replace=False))
        print(f"highD recordings {cli.highd}: {len(idx)} samples, "
              f"classes {np.bincount(ds.D[idx, 9].astype(int), minlength=4)[1:]}")
    else:
        mat = os.path.join(cli.data, f"{cli.split}Set.mat")
        ds = lo.NgsimDataset(mat, t_f=50, d_s=1)
        excl = set()
        for other in cli.exclude_overlap_with:
            excl |= overlapping_tracks(mat, os.path.join(cli.data, f"{other}Set.mat"))
        idx = thinned_indices(ds, cli.gap, exclude_tracks=excl,
                              max_samples=cli.max_samples,
                              one_per_track=cli.one_per_track, seed=cli.seed)
        keys = track_ids(ds)[idx]
        print(f"{cli.split}: {len(idx)} rows from {len(np.unique(keys))} tracks "
              f"(gap {cli.gap}s, excluded {len(excl)} overlapping tracks, shift {cli.shift})")

    feats = extract(ds, idx, gd, gen, device, batch=cli.batch,
                    shift=cli.shift, workers=0 if cli.highd else cli.workers)
    D = ds.D
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    np.savez_compressed(
        cli.out, **feats,
        ds_id=D[idx, 0].astype(np.int32), veh_id=D[idx, 1].astype(np.int32),
        frame=D[idx, 2].astype(np.int32), row_idx=idx.astype(np.int64),
        split=cli.split, gap_s=cli.gap, shift=cli.shift, seed=cli.seed)
    print("saved", cli.out, {k: v.shape for k, v in feats.items()})


if __name__ == "__main__":
    main()
