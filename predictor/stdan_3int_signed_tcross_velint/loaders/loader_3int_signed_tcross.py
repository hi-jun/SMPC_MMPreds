from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as scp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from experiments.stdan_3int_signed_tcross_velint.configs.config_3int_signed_tcross import args


class Ngsim3IntTcrossDataset(Dataset):
    def __init__(self, mat_file: str | Path, t_h=30, t_f=50, d_s=1, enc_size=64, grid_size=(13, 3)):
        mat_file = Path(mat_file).resolve()
        expected_root = Path(args["train_mat"]).resolve().parent
        if expected_root not in mat_file.parents and mat_file.parent != expected_root:
            raise ValueError(f"Refusing to load non-experiment dataset: {mat_file}")

        try:
            mat_data = scp.loadmat(mat_file)
            self.D = mat_data["traj"].astype(np.float32)
            self.T = mat_data["tracks"]
            print(f"Loaded {mat_file} using scipy.io")
        except NotImplementedError:
            print(f"v7.3 mat file detected. Loading {mat_file} using h5py...")
            with h5py.File(mat_file, "r") as f:
                self.D = np.array(f["traj"]).T.astype(np.float32)
                tracks_refs = np.array(f["tracks"]).T
                self.T = np.empty(tracks_refs.shape, dtype=object)
                with tqdm(total=tracks_refs.size, desc="Loading tracks") as pbar:
                    for r in range(tracks_refs.shape[0]):
                        for c in range(tracks_refs.shape[1]):
                            ref = tracks_refs[r, c]
                            try:
                                self.T[r, c] = np.array(f[ref]).T.astype(np.float32)
                            except Exception:
                                self.T[r, c] = np.empty((0, 0), dtype=np.float32)
                            pbar.update(1)

        if self.D.shape[1] < 52:
            raise AssertionError(f"Expected at least 52 columns in new dataset, got {self.D.shape}")
        labels = np.unique(self.D[:, 9]).astype(np.int64)
        if not set(labels.tolist()).issubset({0, 1, 2}):
            raise AssertionError(f"Intention labels must be 0, 1, 2; found {labels}")
        masks = np.unique(self.D[:, 12]).astype(np.int64)
        if not set(masks.tolist()).issubset({0, 1}):
            raise AssertionError(f"t_cross_mask must be binary; found {masks}")

        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.grid_size = grid_size
        self.alltime = 0.0
        self.count = 0

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        frame = self.D[idx, 2]
        grid = self.D[idx, 13:]

        intent_label = int(self.D[idx, 9])
        if intent_label not in (0, 1, 2):
            raise AssertionError(f"Invalid intent label at idx={idx}: {intent_label}")
        intent_enc = np.zeros(args["num_intentions"], dtype=np.float32)
        intent_enc[intent_label] = 1.0
        t_cross_raw = np.float32(self.D[idx, 10])
        t_cross_norm = np.float32(self.D[idx, 11])
        t_cross_mask = np.float32(self.D[idx, 12])

        hist = self.get_history(veh_id, frame, veh_id, ds_id)
        fut = self.get_future(veh_id, frame, ds_id)
        va = self.get_va(veh_id, frame, veh_id, ds_id)
        lane = self.get_lane(veh_id, frame, veh_id, ds_id)
        cclass = self.get_class(veh_id, frame, veh_id, ds_id)
        refdistance = np.zeros((len(hist), 1), dtype=np.float32)

        if fut.shape != (self.t_f // self.d_s, 2):
            raise AssertionError(f"Future shape mismatch at idx={idx}: {fut.shape}")
        if t_cross_mask not in (0.0, 1.0):
            raise AssertionError(f"Invalid t_cross_mask at idx={idx}: {t_cross_mask}")
        max_signed_step = float(args["t_cross_norm_denominator"])
        if t_cross_mask > 0.5 and not (-max_signed_step <= t_cross_raw <= max_signed_step):
            raise AssertionError(f"t_cross_raw must be a signed relative step at idx={idx}: {t_cross_raw}")

        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []
        for nbr_id in grid:
            nbr_id = int(nbr_id)
            nbr_hist = self.get_history(nbr_id, frame, veh_id, ds_id)
            if nbr_hist.shape != (0, 2):
                dist = np.sqrt(np.sum(np.square(hist - nbr_hist), axis=1)).reshape(-1, 1)
            else:
                dist = np.empty((0, 1), dtype=np.float32)
            neighbors.append(nbr_hist)
            neighborsva.append(self.get_va(nbr_id, frame, veh_id, ds_id))
            neighborslane.append(self.get_lane(nbr_id, frame, veh_id, ds_id).reshape(-1, 1))
            neighborsclass.append(self.get_class(nbr_id, frame, veh_id, ds_id).reshape(-1, 1))
            neighborsdistance.append(dist.astype(np.float32))

        return (
            hist, fut, neighbors, intent_enc, intent_label, t_cross_raw, t_cross_norm, t_cross_mask,
            va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass,
        )

    def _track(self, ds_id: int, veh_id: int) -> np.ndarray:
        if veh_id <= 0 or self.T.shape[1] <= veh_id - 1:
            return np.empty((0, 0), dtype=np.float32)
        track = self.T[ds_id - 1][veh_id - 1]
        if track.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        return track.transpose().astype(np.float32)

    def _history_slice(self, veh_id, frame, ref_veh_id, ds_id, cols):
        if veh_id == 0:
            return np.empty((0, len(np.atleast_1d(cols))), dtype=np.float32)
        ref_track = self._track(ds_id, ref_veh_id)
        veh_track = self._track(ds_id, veh_id)
        if ref_track.size == 0 or veh_track.size == 0:
            return np.empty((0, len(np.atleast_1d(cols))), dtype=np.float32)
        ref_hits = np.where(ref_track[:, 0] == frame)[0]
        veh_hits = np.where(veh_track[:, 0] == frame)[0]
        if ref_hits.size == 0 or veh_hits.size == 0:
            return np.empty((0, len(np.atleast_1d(cols))), dtype=np.float32)
        current = int(veh_hits[0])
        stpt = max(0, current - self.t_h)
        enpt = current + 1
        hist = veh_track[stpt:enpt:self.d_s, cols]
        if hist.ndim == 1:
            hist = hist.reshape(-1, 1)
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty((0, hist.shape[1]), dtype=np.float32)
        return hist.astype(np.float32)

    def get_lane(self, veh_id, frame, ref_veh_id, ds_id):
        return self._history_slice(veh_id, frame, ref_veh_id, ds_id, 5).reshape(-1)

    def get_class(self, veh_id, frame, ref_veh_id, ds_id):
        return self._history_slice(veh_id, frame, ref_veh_id, ds_id, 6).reshape(-1)

    def get_va(self, veh_id, frame, ref_veh_id, ds_id):
        return self._history_slice(veh_id, frame, ref_veh_id, ds_id, slice(3, 5))

    def get_history(self, veh_id, frame, ref_veh_id, ds_id):
        if veh_id == 0:
            return np.empty((0, 2), dtype=np.float32)
        ref_track = self._track(ds_id, ref_veh_id)
        hist = self._history_slice(veh_id, frame, ref_veh_id, ds_id, slice(1, 3))
        if hist.shape != (self.t_h // self.d_s + 1, 2):
            return np.empty((0, 2), dtype=np.float32)
        ref_hits = np.where(ref_track[:, 0] == frame)[0]
        ref_pos = ref_track[ref_hits[0], 1:3]
        return (hist - ref_pos).astype(np.float32)

    def get_future(self, veh_id, frame, ds_id):
        veh_track = self._track(ds_id, veh_id)
        hits = np.where(veh_track[:, 0] == frame)[0]
        if hits.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        current = int(hits[0])
        ref_pos = veh_track[current, 1:3]
        stpt = current + self.d_s
        enpt = min(len(veh_track), current + self.t_f + 1)
        return (veh_track[stpt:enpt:self.d_s, 1:3] - ref_pos).astype(np.float32)

    def collate_fn(self, samples):
        t0 = time.time()
        nbr_batch_size = sum(sum(len(nbrs[i]) != 0 for i in range(len(nbrs))) for _, _, nbrs, *_ in samples)
        maxlen = self.t_h // self.d_s + 1
        futlen = self.t_f // self.d_s

        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size).bool()
        map_position = torch.zeros(0, 2)

        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(futlen, len(samples), 2)
        op_mask_batch = torch.zeros(futlen, len(samples), 2)
        intent_enc_batch = torch.zeros(len(samples), args["num_intentions"])
        intent_label_batch = torch.zeros(len(samples), dtype=torch.long)
        t_cross_raw_batch = torch.zeros(len(samples), 1)
        t_cross_norm_batch = torch.zeros(len(samples), 1)
        t_cross_mask_batch = torch.zeros(len(samples), 1)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        distance_batch = torch.zeros(maxlen, len(samples), 1)

        counts = [0, 0, 0, 0, 0]
        for sample_id, sample in enumerate(samples):
            (
                hist, fut, nbrs, intent_enc, intent_label, t_cross_raw, t_cross_norm, t_cross_mask,
                va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass,
            ) = sample

            hist_batch[:len(hist), sample_id, :] = torch.from_numpy(hist)
            fut_batch[:len(fut), sample_id, :] = torch.from_numpy(fut)
            op_mask_batch[:len(fut), sample_id, :] = 1
            intent_enc_batch[sample_id, :] = torch.from_numpy(intent_enc)
            intent_label_batch[sample_id] = int(intent_label)
            t_cross_raw_batch[sample_id, 0] = float(t_cross_raw)
            t_cross_norm_batch[sample_id, 0] = float(t_cross_norm)
            t_cross_mask_batch[sample_id, 0] = float(t_cross_mask)
            va_batch[:len(va), sample_id, :] = torch.from_numpy(va)
            lane_batch[:len(lane), sample_id, 0] = torch.from_numpy(lane)
            class_batch[:len(cclass), sample_id, 0] = torch.from_numpy(cclass)
            distance_batch[:len(refdistance), sample_id, :] = torch.from_numpy(refdistance)

            for grid_id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[:len(nbr), counts[0], :] = torch.from_numpy(nbr)
                    x = grid_id % self.grid_size[0]
                    y = grid_id // self.grid_size[0]
                    mask_batch[sample_id, y, x, :] = True
                    map_position = torch.cat((map_position, torch.tensor([[y, x]])), 0)
                    counts[0] += 1
            for nbrva in neighborsva:
                if len(nbrva) != 0:
                    nbrsva_batch[:len(nbrva), counts[1], :] = torch.from_numpy(nbrva)
                    counts[1] += 1
            for nbrlane in neighborslane:
                if len(nbrlane) != 0:
                    nbrslane_batch[:len(nbrlane), counts[2], :] = torch.from_numpy(nbrlane)
                    counts[2] += 1
            for nbrdis in neighborsdistance:
                if len(nbrdis) != 0:
                    nbrsdis_batch[:len(nbrdis), counts[3], :] = torch.from_numpy(nbrdis)
                    counts[3] += 1
            for nbrclass in neighborsclass:
                if len(nbrclass) != 0:
                    nbrsclass_batch[:len(nbrclass), counts[4], :] = torch.from_numpy(nbrclass)
                    counts[4] += 1

        self.alltime += time.time() - t0
        self.count += args["num_worker"]

        return (
            hist_batch, nbrs_batch, mask_batch, intent_enc_batch, intent_label_batch,
            t_cross_raw_batch, t_cross_norm_batch, t_cross_mask_batch, fut_batch, op_mask_batch,
            va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch,
            class_batch, nbrsclass_batch, map_position,
        )
