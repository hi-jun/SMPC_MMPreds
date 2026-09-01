from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "stdan_3int_signed_tcross_velint"
RAW_ROOT = REPO_ROOT / "stdan" / "data"
OUT_ROOT = EXP_ROOT / "data" / "processed_3int_signed_tcross"

RAW_FILES = [
    "trajectories-0750am-0805am.txt",
    "trajectories-0805am-0820am.txt",
    "trajectories-0820am-0835am.txt",
    "trajectories-0400-0415.txt",
    "trajectories-0500-0515.txt",
    "trajectories-0515-0530.txt",
]

HIST_FRAMES = 30
FUTURE_FRAMES = 50
LAT_LABEL_WINDOW = 28
TCROSS_NORM_DENOMINATOR = float(LAT_LABEL_WINDOW)
LANE_WIDTH_FT = 12.0
GRID_CELLS = 39
GRID_START_COL = 13

# Output traj columns:
# 0 ds_id, 1 veh_id, 2 frame_id, 3 local_x, 4 local_y, 5 v_vel,
# 6 v_acc, 7 lane_id, 8 v_class, 9 intent_label, 10 t_cross_signed_raw,
# 11 t_cross_signed_norm, 12 t_cross_mask, 13:52 social-grid vehicle ids.


@dataclass
class DatasetPack:
    rows_by_vehicle: Dict[Tuple[int, int], np.ndarray]
    frame_maps: Dict[int, Dict[int, np.ndarray]]
    tracks: np.ndarray
    max_veh_id: int


def load_raw_dataset(ds_id: int, path: Path, max_rows: int | None) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float32, max_rows=max_rows)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    selected = np.empty((raw.shape[0], 13), dtype=np.float32)
    selected[:, 0] = ds_id
    selected[:, 1] = raw[:, 0]   # Vehicle_ID
    selected[:, 2] = raw[:, 1]   # Frame_ID
    selected[:, 3] = raw[:, 4]   # Local_X
    selected[:, 4] = raw[:, 5]   # Local_Y
    selected[:, 5] = raw[:, 11]  # v_Vel
    selected[:, 6] = raw[:, 12]  # v_Acc
    selected[:, 7] = raw[:, 13]  # Lane_ID
    selected[:, 8] = raw[:, 10]  # v_Class

    if ds_id <= 3:
        selected[selected[:, 7] >= 6, 7] = 6

    return selected


def label_vehicle_rows(rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows[:, 2], kind="mergesort")
    rows = rows[order].copy()
    lanes = rows[:, 7]
    n = rows.shape[0]
    idx = np.arange(n)

    intent = np.zeros(n, dtype=np.float32)
    t_cross_raw = np.zeros(n, dtype=np.float32)
    t_cross_mask = np.zeros(n, dtype=np.float32)

    transition_idx = np.where(lanes[1:] != lanes[:-1])[0] + 1
    for transition in transition_idx:
        lane_delta = lanes[transition] - lanes[transition - 1]
        if lane_delta == 0:
            continue
        label = 1.0 if lane_delta < 0 else 2.0
        start = max(0, transition - LAT_LABEL_WINDOW)
        end = min(n, transition + LAT_LABEL_WINDOW + 1)
        window_idx = np.arange(start, end)
        signed_offsets = transition - window_idx

        unset = t_cross_mask[window_idx] < 0.5
        closer = np.abs(signed_offsets) < np.abs(t_cross_raw[window_idx])
        choose = unset | closer
        if np.any(choose):
            chosen_idx = window_idx[choose]
            intent[chosen_idx] = label
            t_cross_raw[chosen_idx] = signed_offsets[choose].astype(np.float32)
            t_cross_mask[chosen_idx] = 1.0

    t_cross_norm = t_cross_raw / TCROSS_NORM_DENOMINATOR

    rows[:, 9] = intent
    rows[:, 10] = t_cross_raw
    rows[:, 11] = t_cross_norm
    rows[:, 12] = t_cross_mask
    return rows


def build_base_dataset(max_rows_per_file: int | None) -> Tuple[List[np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    datasets: List[np.ndarray] = []
    rows_by_vehicle: Dict[Tuple[int, int], np.ndarray] = {}

    for ds_id, name in enumerate(RAW_FILES, start=1):
        path = RAW_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Missing raw NGSIM file: {path}")
        print(f"Loading {path}")
        data = load_raw_dataset(ds_id, path, max_rows_per_file)
        labeled_chunks = []
        for veh_id in np.unique(data[:, 1]).astype(np.int64):
            veh_rows = data[data[:, 1] == veh_id]
            if veh_rows.shape[0] == 0:
                continue
            labeled = label_vehicle_rows(veh_rows)
            rows_by_vehicle[(ds_id, int(veh_id))] = labeled
            labeled_chunks.append(labeled)
        ds_rows = np.vstack(labeled_chunks).astype(np.float32)
        ds_rows = ds_rows[np.lexsort((ds_rows[:, 2], ds_rows[:, 1]))]
        datasets.append(ds_rows)
        print(f"  dataset {ds_id}: {ds_rows.shape[0]} rows, {len(labeled_chunks)} vehicles")

    return datasets, rows_by_vehicle


def build_frame_maps(rows_by_vehicle: Dict[Tuple[int, int], np.ndarray]) -> Dict[int, Dict[int, np.ndarray]]:
    chunks_by_ds: Dict[int, List[np.ndarray]] = {}
    for (ds_id, _), rows in rows_by_vehicle.items():
        chunks_by_ds.setdefault(ds_id, []).append(rows)

    frame_maps: Dict[int, Dict[int, np.ndarray]] = {}
    for ds_id, chunks in chunks_by_ds.items():
        all_rows = np.vstack(chunks)
        order = np.argsort(all_rows[:, 2], kind="mergesort")
        all_rows = all_rows[order]
        frames: Dict[int, np.ndarray] = {}
        unique_frames, starts = np.unique(all_rows[:, 2].astype(np.int64), return_index=True)
        ends = np.r_[starts[1:], all_rows.shape[0]]
        for frame, start, end in zip(unique_frames, starts, ends):
            frames[int(frame)] = all_rows[start:end]
        frame_maps[ds_id] = frames
    return frame_maps


def build_tracks(rows_by_vehicle: Dict[Tuple[int, int], np.ndarray]) -> Tuple[np.ndarray, int]:
    max_ds = max(ds_id for ds_id, _ in rows_by_vehicle)
    max_veh_id = max(veh_id for _, veh_id in rows_by_vehicle)
    tracks = np.empty((max_ds, max_veh_id), dtype=object)
    for idx in np.ndindex(tracks.shape):
        tracks[idx] = np.empty((0, 0), dtype=np.float32)

    for (ds_id, veh_id), rows in rows_by_vehicle.items():
        # frame, local_x, local_y, v_vel, v_acc, lane_id, v_class,
        # intent_label, t_cross_raw, t_cross_norm, t_cross_mask
        tracks[ds_id - 1, veh_id - 1] = rows[:, 2:13].T.astype(np.float32)
    return tracks, max_veh_id


def filtered_samples(datasets: Iterable[np.ndarray]) -> np.ndarray:
    filtered = []
    for data in datasets:
        for veh_id in np.unique(data[:, 1]).astype(np.int64):
            veh = data[data[:, 1] == veh_id]
            if veh.shape[0] > HIST_FRAMES + FUTURE_FRAMES:
                filtered.append(veh[HIST_FRAMES: -FUTURE_FRAMES])
    if not filtered:
        return np.empty((0, 13), dtype=np.float32)
    return np.vstack(filtered).astype(np.float32)


def split_by_dataset(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = [], [], []
    for ds_id in range(1, 7):
        rows = samples[samples[:, 0] == ds_id]
        len1 = int(round(rows.shape[0] * 0.7))
        len2 = int(round(rows.shape[0] * 0.8))
        train.append(rows[:len1])
        val.append(rows[len1:len2])
        test.append(rows[len2:])
    return np.vstack(train), np.vstack(val), np.vstack(test)


def balance_211(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx_lk = np.where(rows[:, 9] == 0)[0]
    idx_llc = np.where(rows[:, 9] == 1)[0]
    idx_rlc = np.where(rows[:, 9] == 2)[0]
    target = min(len(idx_lk) // 2, len(idx_llc), len(idx_rlc))
    if target == 0:
        raise RuntimeError(
            f"Cannot balance set: LK={len(idx_lk)}, LLC={len(idx_llc)}, RLC={len(idx_rlc)}"
        )
    chosen = np.concatenate([
        rng.choice(idx_lk, 2 * target, replace=False),
        rng.choice(idx_llc, target, replace=False),
        rng.choice(idx_rlc, target, replace=False),
    ])
    out = rows[chosen]
    order = np.lexsort((out[:, 2], out[:, 1], out[:, 0]))
    return out[order].astype(np.float32)


def compute_grid_for_rows(rows: np.ndarray, frame_maps: Dict[int, Dict[int, np.ndarray]]) -> np.ndarray:
    out = np.zeros((rows.shape[0], GRID_CELLS), dtype=np.float32)
    for r, row in enumerate(rows):
        ds_id = int(row[0])
        veh_id = int(row[1])
        frame = int(row[2])
        lane = row[7]
        y0 = row[4]
        frame_rows = frame_maps[ds_id].get(frame)
        if frame_rows is None:
            continue
        for lane_offset, base in [(-1, 0), (0, 13), (1, 26)]:
            lane_rows = frame_rows[frame_rows[:, 7] == lane + lane_offset]
            for nbr in lane_rows:
                nbr_id = int(nbr[1])
                if nbr_id == veh_id:
                    continue
                dy = nbr[4] - y0
                if abs(dy) < 90.0:
                    grid_ind = int(1 + np.round((dy + 90.0) / 15.0))
                    if 1 <= grid_ind <= 13:
                        out[r, base + grid_ind - 1] = float(nbr_id)
    return out


def attach_grid(rows: np.ndarray, frame_maps: Dict[int, Dict[int, np.ndarray]]) -> np.ndarray:
    grid = compute_grid_for_rows(rows, frame_maps)
    return np.hstack([rows, grid]).astype(np.float32)


def augment_tracks(
    rows_by_vehicle: Dict[Tuple[int, int], np.ndarray],
    tracks: np.ndarray,
    max_veh_id: int,
    frame_shift: int,
) -> np.ndarray:
    augmented = np.empty((tracks.shape[0], max_veh_id * 2), dtype=object)
    for idx in np.ndindex(augmented.shape):
        augmented[idx] = np.empty((0, 0), dtype=np.float32)
    augmented[:, :max_veh_id] = tracks

    for (ds_id, veh_id), rows in rows_by_vehicle.items():
        new_rows = rows.copy()
        new_rows[:, 1] += max_veh_id
        new_rows[:, 2] += frame_shift
        max_lane = 5 if ds_id <= 3 else 6
        road_width = max_lane * LANE_WIDTH_FT
        new_rows[:, 3] = road_width - new_rows[:, 3]
        new_rows[:, 7] = (max_lane + 1) - new_rows[:, 7]
        left = new_rows[:, 9] == 1
        right = new_rows[:, 9] == 2
        new_rows[left, 9] = 2
        new_rows[right, 9] = 1
        augmented[ds_id - 1, veh_id + max_veh_id - 1] = new_rows[:, 2:13].T.astype(np.float32)

    return augmented


def augment_rows(rows_with_grid: np.ndarray, max_veh_id: int, frame_shift: int) -> np.ndarray:
    aug = rows_with_grid.copy()
    aug[:, 2] += frame_shift
    aug[:, 1] += max_veh_id
    grid_cols = aug[:, GRID_START_COL:].copy()
    has_neighbor = grid_cols > 0
    grid_cols[has_neighbor] += max_veh_id
    aug[:, GRID_START_COL:] = grid_cols

    us101 = aug[:, 0] <= 3
    i80 = aug[:, 0] >= 4
    aug[us101, 7] = 6 - aug[us101, 7]
    aug[us101, 3] = (5 * LANE_WIDTH_FT) - aug[us101, 3]
    aug[i80, 7] = 7 - aug[i80, 7]
    aug[i80, 3] = (6 * LANE_WIDTH_FT) - aug[i80, 3]

    left = aug[:, 9] == 1
    right = aug[:, 9] == 2
    aug[left, 9] = 2
    aug[right, 9] = 1

    left_grid = aug[:, GRID_START_COL:GRID_START_COL + 13].copy()
    right_grid = aug[:, GRID_START_COL + 26:GRID_START_COL + 39].copy()
    aug[:, GRID_START_COL:GRID_START_COL + 13] = right_grid
    aug[:, GRID_START_COL + 26:GRID_START_COL + 39] = left_grid

    valid = (aug[:, 7] > 0) & (aug[:, 7] <= 6)
    return aug[valid].astype(np.float32)


def stats_for_rows(rows: np.ndarray) -> dict:
    intent = rows[:, 9].astype(np.int64)
    mask = rows[:, 12] > 0.5
    lc = intent != 0
    invalid_lc = int(np.sum(lc & ~mask))
    valid_cross = rows[mask, 10]
    hist = Counter(valid_cross.astype(np.int64).tolist())
    lc_count = int(np.sum(lc))
    return {
        "num_samples": int(rows.shape[0]),
        "class_distribution": {
            "LK": int(np.sum(intent == 0)),
            "LLC": int(np.sum(intent == 1)),
            "RLC": int(np.sum(intent == 2)),
        },
        "valid_t_cross_count": int(np.sum(mask)),
        "valid_t_cross_ratio": float(np.mean(mask)) if rows.shape[0] else 0.0,
        "valid_t_cross_distribution": {str(k): int(v) for k, v in sorted(hist.items())},
        "valid_t_cross_min": float(np.min(valid_cross)) if valid_cross.size else None,
        "valid_t_cross_mean": float(np.mean(valid_cross)) if valid_cross.size else None,
        "valid_t_cross_max": float(np.max(valid_cross)) if valid_cross.size else None,
        "invalid_or_ambiguous_lc_count": invalid_lc,
        "valid_t_cross_ratio_among_lc": float(np.sum(mask & lc) / lc_count) if lc_count else 0.0,
        "pre_cross_count": int(np.sum(valid_cross > 0)),
        "at_cross_count": int(np.sum(valid_cross == 0)),
        "post_cross_count": int(np.sum(valid_cross < 0)),
    }


def assert_dataset(rows: np.ndarray) -> None:
    labels = set(np.unique(rows[:, 9]).astype(np.int64).tolist())
    if not labels.issubset({0, 1, 2}):
        raise AssertionError(f"Invalid labels found: {labels}")
    mask_values = set(np.unique(rows[:, 12]).astype(np.int64).tolist())
    if not mask_values.issubset({0, 1}):
        raise AssertionError(f"Invalid t_cross_mask values found: {mask_values}")
    valid = rows[:, 12] > 0.5
    if np.any((rows[valid, 10] < -LAT_LABEL_WINDOW) | (rows[valid, 10] > LAT_LABEL_WINDOW)):
        raise AssertionError("Valid t_cross_raw values must be signed relative steps in [-28, 28]")
    if np.any(rows[rows[:, 9] == 0, 12] != 0):
        raise AssertionError("LK samples must not contribute to t_cross loss")


def save_set(path: Path, rows: np.ndarray, tracks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"traj": rows.astype(np.float32), "tracks": tracks}, do_compression=False)


def run(max_rows_per_file: int | None, output_suffix: str) -> None:
    rng = np.random.default_rng(72)
    out_root = OUT_ROOT if output_suffix == "" else OUT_ROOT / output_suffix
    out_root.mkdir(parents=True, exist_ok=True)

    datasets, rows_by_vehicle = build_base_dataset(max_rows_per_file)
    frame_maps = build_frame_maps(rows_by_vehicle)
    tracks, max_veh_id = build_tracks(rows_by_vehicle)
    all_filtered = filtered_samples(datasets)
    if all_filtered.size == 0:
        raise RuntimeError("No samples survived history/future filtering.")

    global_max_frame = int(np.max(all_filtered[:, 2]))
    min_aug_frame = int(np.min(all_filtered[:, 2]))
    frame_shift = global_max_frame - min_aug_frame + 100

    train, val, test = split_by_dataset(all_filtered)
    train = balance_211(train, rng)
    val = balance_211(val, rng)
    test = balance_211(test, rng)

    print("Computing social grids for balanced sets...")
    train = attach_grid(train, frame_maps)
    val = attach_grid(val, frame_maps)
    test = attach_grid(test, frame_maps)

    tracks_aug = augment_tracks(rows_by_vehicle, tracks, max_veh_id, frame_shift)
    final_sets = {
        "TrainSet": np.vstack([train, augment_rows(train, max_veh_id, frame_shift)]).astype(np.float32),
        "ValSet": np.vstack([val, augment_rows(val, max_veh_id, frame_shift)]).astype(np.float32),
        "TestSet": np.vstack([test, augment_rows(test, max_veh_id, frame_shift)]).astype(np.float32),
    }

    stats = {
        "raw_data_path": str(RAW_ROOT),
        "output_path": str(out_root),
        "intent_mapping": {"0": "LK", "1": "LLC", "2": "RLC"},
        "t_cross_definition": "signed relative lane-transition time: transition_index - current_index within +/-28 frames",
        "t_cross_norm_formula": f"t_cross_signed_raw / {TCROSS_NORM_DENOMINATOR:g}",
        "frame_shift_for_augmentation": frame_shift,
        "dropped_or_invalid_samples": 0,
    }

    for name, rows in final_sets.items():
        assert_dataset(rows)
        stats[name] = stats_for_rows(rows)
        save_set(out_root / f"{name}.mat", rows, tracks_aug)
        print(f"{name}: {rows.shape}")

    with (out_root / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved stats to {out_root / 'dataset_stats.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run on a small raw-file subset.")
    parser.add_argument("--full", action="store_true", help="Generate the full processed dataset.")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        run(max_rows_per_file=args.max_rows_per_file or 200000, output_suffix="smoke")
    elif args.full:
        run(max_rows_per_file=args.max_rows_per_file, output_suffix="")
    else:
        raise SystemExit("Pass --smoke or --full")


if __name__ == "__main__":
    main()
