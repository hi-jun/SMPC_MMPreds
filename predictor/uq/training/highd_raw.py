"""Build STDAN encoder inputs directly from the raw highD CSVs.

The highD copies that were already preprocessed into the NGSIM `.mat` layout
cannot be used: their social grid does not describe the recording. Measured on
`only_interaction_drivedirection2`, the grid is 25-39 of 39 cells occupied with
own-lane vehicles 2.2 m apart at 27 m/s, while the raw tracks give a median
headway of 53.5 m and a median time headway of 1.86 s; the stored `frame` index
also fails to locate the target vehicle in the source CSV. So this module goes
back to the released `data/*_tracks.csv` and reproduces the NGSIM pipeline
itself, which keeps the comparison honest: the same loader-format 14-tuple, the
same feet-based units, the same 13x3 grid geometry, the same maneuver labels.

Conventions matched to NGSIM (verified in loader2_hdf5_vel / the grid probe):
  hist[:, 0]  lateral, positive to the RIGHT of travel     [ft]
  hist[:, 1]  longitudinal, positive ahead                 [ft]
  va[:, 0]    speed, va[:, 1] acceleration                 [ft/s, ft/s^2]
  grid        13 longitudinal x 3 lateral cells, +-90 ft, cell id = col*13 + lon
  lat_enc     1 = keep, 2 = left change, 3 = right change  (D[:, 9] convention)

highD records both directions in one frame with the image y axis pointing down.
Vehicles with xVelocity < 0 drive in -x; mirroring x and y for those puts every
vehicle in a single "driving +x, right hand side = +y" frame, so both
directions are usable and the lateral sign matches NGSIM.

25 Hz is resampled to 10 Hz by linear interpolation on the frame index (0.1 s =
2.5 raw frames), which avoids the 2/3-frame stutter of nearest-frame picking.
"""
import os

import numpy as np
import pandas as pd

from config_vel import args
from loader2_hdf5_vel import NgsimDataset

M2FT = 1.0 / 0.3048
DATA_DIR = "/home/core-dev/HJ/dataset/raw_data_highD/raw_highD/data"
LANE_W_M = 3.75          # highD lane markings are ~3.7-3.8 m apart
LON_RANGE_FT = 90.0      # the NGSIM grid reach
GRID = (13, 3)


class HighDRaw(NgsimDataset):
    """NgsimDataset-compatible view of the raw highD recordings.

    Only __init__ and __getitem__ are new; collate_fn and the rest are
    inherited so that extract_features.extract() runs unchanged.
    """

    def __init__(self, recordings, t_h=30, t_f=50, d_s=1, enc_size=64,
                 grid_size=GRID, data_dir=DATA_DIR, stride_s=1.0,
                 label_window_s=4.0, max_per_recording=None, seed=0):
        self.t_h, self.t_f, self.d_s = t_h, t_f, d_s
        self.enc_size, self.grid_size = enc_size, grid_size
        self.data_dir = data_dir
        self.label_window = label_window_s
        self.hz = 10.0
        self.alltime = 0.0     # collate_fn (inherited) keeps timing counters
        self.count = 0
        self._tracks = {}          # rec -> {vid: interpolated arrays}
        self._frames = {}          # rec -> {t10: DataFrame view}
        rows = []
        rng = np.random.RandomState(seed)
        for rec in recordings:
            tab = self._load_recording(rec)
            self._tracks[rec] = tab
            cand = []
            for vid, tr in tab.items():
                n = len(tr["t"])
                # need t_h past steps and label_window future steps
                lo = t_h
                hi = n - int(label_window_s * self.hz) - 1
                for k in range(lo, max(lo, hi), max(1, int(stride_s * self.hz))):
                    cand.append((rec, vid, k))
            if max_per_recording and len(cand) > max_per_recording:
                cand = [cand[i] for i in
                        rng.choice(len(cand), max_per_recording, replace=False)]
            rows += cand
        self.rows = rows
        # D mirrors the NGSIM traj columns the samplers use: dsId, vehId, frame,
        # ... , lat class at col 9. Only those columns are populated.
        self.D = np.zeros((len(rows), 50))
        for i, (rec, vid, k) in enumerate(rows):
            tr = self._tracks[rec][vid]
            self.D[i, 0] = rec
            self.D[i, 1] = vid
            self.D[i, 2] = tr["t"][k]
            self.D[i, 9] = self._lat_class(tr, k)
            self.D[i, 10] = 2      # longitudinal class unused here

    # ------------------------------------------------------------- loading

    def _load_recording(self, rec):
        path = os.path.join(self.data_dir, f"{rec:02d}_tracks.csv")
        df = pd.read_csv(path, usecols=["frame", "id", "x", "y", "xVelocity",
                                        "yVelocity", "xAcceleration", "laneId",
                                        "width", "height"])
        # single travel frame: mirror the vehicles heading -x
        sign = np.where(df["xVelocity"].values >= 0, 1.0, -1.0)
        df["X"] = df["x"].values * sign
        df["Y"] = df["y"].values * sign
        df["V"] = df["xVelocity"].values * sign
        df["A"] = df["xAcceleration"].values * sign
        df["VY"] = df["yVelocity"].values * sign

        out = {}
        for vid, g in df.groupby("id"):
            g = g.sort_values("frame")
            f = g["frame"].values.astype(float)
            if len(f) < self.t_h * 3:
                continue
            # 25 Hz -> 10 Hz on a GLOBAL phase: every vehicle must land on the
            # same instants or the neighbour lookup never finds a match
            # (per-vehicle phases would each start at their own first frame).
            t10 = np.arange(np.ceil(f[0] / 2.5) * 2.5, f[-1] + 1e-9, 2.5)
            if len(t10) < self.t_h * 2:
                continue
            it = lambda col: np.interp(t10, f, g[col].values)
            out[vid] = {"t": t10, "X": it("X"), "Y": it("Y"), "V": it("V"),
                        "A": it("A"), "lane": np.interp(t10, f, g["laneId"].values),
                        "len": float(g["width"].iloc[0]),
                        "dir": float(np.sign(g["xVelocity"].mean()))}
        # per 10 Hz step index: which vehicles exist, for the neighbour search
        idx = {}
        for vid, tr in out.items():
            for j, t in enumerate(tr["t"]):
                idx.setdefault(round(t, 3), []).append((vid, j))
        self._frames[rec] = idx
        return out

    # ------------------------------------------------------------- labels

    def _lat_class(self, tr, k):
        """1 keep / 2 left change / 3 right change, from the lane index within
        +-label_window seconds - the same rule NGSIM's preprocessing uses."""
        w = int(self.label_window * self.hz)
        a = tr["lane"][max(0, k - w)]
        b = tr["lane"][min(len(tr["lane"]) - 1, k + w)]
        # highD lane ids grow downward in the image; after mirroring, a vehicle
        # moving to its RIGHT increases Y, which is the same direction the lane
        # id moves for lower-lane (dir +1) vehicles and the opposite for upper.
        d = (b - a) * (1.0 if tr["dir"] >= 0 else -1.0)
        if d > 0.5:
            return 3          # right
        if d < -0.5:
            return 2          # left
        return 1

    # ------------------------------------------------------------- sample

    def __len__(self):
        return len(self.rows)

    def _history(self, tr, k, ox, oy):
        """t_h+1 steps of [lateral, longitudinal] in feet, target-relative."""
        s = slice(k - self.t_h, k + 1)
        lat = (tr["Y"][s] - oy) * M2FT
        lon = (tr["X"][s] - ox) * M2FT
        return np.stack([lat, lon], axis=1)

    def __getitem__(self, i):
        rec, vid, k = self.rows[i]
        tab = self._tracks[rec]
        tr = tab[vid]
        ox, oy, t_now = tr["X"][k], tr["Y"][k], round(tr["t"][k], 3)

        hist = self._history(tr, k, ox, oy)
        va = np.stack([tr["V"][k - self.t_h:k + 1] * M2FT,
                       tr["A"][k - self.t_h:k + 1] * M2FT], axis=1)
        lane = tr["lane"][k - self.t_h:k + 1].astype(float)

        # ---- neighbours into the 13 x 3 grid, NGSIM cell geometry
        cell_lon = 2.0 * LON_RANGE_FT / self.grid_size[0]
        lat_edges = np.array([-1.5, -0.5, 0.5, 1.5]) * LANE_W_M
        neighbors = [[] for _ in range(self.grid_size[0] * self.grid_size[1])]
        nbrsva = [[] for _ in neighbors]
        nbrslane = [[] for _ in neighbors]
        nbrsdis = [[] for _ in neighbors]
        nbrscls = [[] for _ in neighbors]
        for other, j in self._frames[rec].get(t_now, ()):
            if other == vid:
                continue
            o = tab[other]
            if o["dir"] != tr["dir"]:
                continue                    # oncoming traffic is not a neighbour
            dlon = (o["X"][j] - ox) * M2FT
            dlat = (o["Y"][j] - oy)
            if abs(dlon) > LON_RANGE_FT or not (lat_edges[0] <= dlat <= lat_edges[-1]):
                continue
            if j < self.t_h:
                continue
            g_lat = int(np.digitize(dlat, lat_edges) - 1)
            g_lon = int(np.clip((dlon + LON_RANGE_FT) // cell_lon, 0,
                                self.grid_size[0] - 1))
            cid = g_lat * self.grid_size[0] + g_lon
            if neighbors[cid]:                     # one vehicle per cell
                continue
            neighbors[cid] = self._history(o, j, ox, oy)
            nbrsva[cid] = np.stack([o["V"][j - self.t_h:j + 1] * M2FT,
                                    o["A"][j - self.t_h:j + 1] * M2FT], axis=1)
            nbrslane[cid] = o["lane"][j - self.t_h:j + 1].reshape(-1, 1)
            nbrsdis[cid] = np.full((self.t_h + 1, 1), abs(dlon))
            nbrscls[cid] = np.full((self.t_h + 1, 1), 2.0)

        lat_enc = np.zeros(args["lat_length"]); lat_enc[int(self.D[i, 9]) - 1] = 1
        lon_enc = np.zeros(args["lon_length"]); lon_enc[1] = 1
        n_f = self.t_f // self.d_s
        fut_vel = np.zeros((n_f, 2)); fut_pos = np.zeros((n_f, 2))
        refdis = np.zeros((self.t_h + 1, 1))
        cclass = np.full(self.t_h + 1, 2.0)
        return (hist, fut_vel, fut_pos, neighbors, lat_enc, lon_enc, va, nbrsva,
                lane, nbrslane, refdis, nbrsdis, cclass, nbrscls)
