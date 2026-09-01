from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from predictor.stdan.stdan_predictor import FT2M, transform_points
from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter


DEFAULT_CKPT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "0113_ratio211"
    / "best_model.pt"
)


def _default_args(
    history: float,
    future: float,
    dt: float,
    device: Optional[str],
) -> dict:
    torch_device = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return {
        "device": torch_device,
        "lstm_encoder_size": 64,
        "n_head": 4,
        "att_out": 48,
        "in_length": int(round(float(history) / float(dt))) + 1,
        "out_length": int(round(float(future) / float(dt))),
        "f_length": 5,
        "traj_linear_hidden": 32,
        "use_elu": True,
        "dropout": 0.5,
        "relu": 0.1,
        "lat_length": 3,
        "lon_length": 3,
        "use_true_man": False,
        "use_spatial": False,
        "intention_weight": 1.0,
        "use_maneuvers": True,
        "cat_pred": True,
        "use_mse": False,
        "train_flag": False,
    }


def _normalized_prob(tensor: torch.Tensor, expected: int) -> np.ndarray:
    prob = tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if prob.size != expected:
        raise ValueError(f"Expected {expected} probabilities, got {prob.size}")
    total = float(np.sum(prob))
    if total <= 1.0e-12:
        return np.full(expected, 1.0 / expected, dtype=np.float64)
    return prob / total


class STDANVelACCAdapter(STDAN3IntACCAdapter):
    """
    ACC adapter for the legacy STDAN velocity model.

    The checkpoint emits 9 trajectories ordered by longitudinal mode first and
    lateral mode second: index = lon_idx * 3 + lat_idx. ACC consumes 3 lateral
    hypotheses, so this adapter keeps only the Constant-speed longitudinal mode
    (lon_idx=0) for each lateral mode. raw mode probabilities come from lat_pred
    in LK/LLC/RLC order.
    """

    mode_names = ["LK", "LLC", "RLC"]
    longitudinal_mode_names = ["Constant", "Deceleration", "Acceleration"]

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        history: float = 3.0,
        future: float = 5.0,
        dt: float = 0.1,
        device: Optional[str] = None,
        mc_dropout: bool = False,
        load_model: bool = True,
        pad_short_history: bool = True,
        min_history_samples: int = 3,
        longitudinal_mode_idx: int = 0,
    ):
        self.args = _default_args(history, future, dt, device)
        self.dt = float(dt)
        self.history = float(history)
        self.future = float(future)
        self.device = self.args["device"]
        self.in_length = int(self.args["in_length"])
        self.out_length = int(self.args["out_length"])
        self.full_length = self.in_length + 2
        self.encoder_size = int(self.args["lstm_encoder_size"])
        self.grid_size = (13, 3)
        self.mc_dropout = bool(mc_dropout)
        self.pad_short_history = bool(pad_short_history)
        self.min_history_samples = int(min_history_samples)
        self.longitudinal_mode_idx = int(longitudinal_mode_idx)
        if not 0 <= self.longitudinal_mode_idx < int(self.args["lon_length"]):
            raise ValueError(
                f"longitudinal_mode_idx must be in [0, {self.args['lon_length'] - 1}]"
            )
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
        if not self.ckpt_path.is_absolute():
            root = Path(__file__).resolve().parents[2]
            self.ckpt_path = root / self.ckpt_path
        self.gdEncoder = None
        self.generator = None
        if load_model:
            self._load_model(self.ckpt_path)

    def _load_model(self, ckpt_path: Path) -> None:
        from predictor.stdan_vel.model5f_mult_vel import GDEncoder, Generator

        if not ckpt_path.exists():
            raise FileNotFoundError(f"STDAN velocity checkpoint not found: {ckpt_path}")
        try:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)
        self.gdEncoder = GDEncoder(self.args).to(self.device)
        self.generator = Generator(self.args).to(self.device)
        self.gdEncoder.load_state_dict(checkpoint["gdEncoder"])
        self.generator.load_state_dict(checkpoint["generator"])
        self.gdEncoder.eval()
        self.generator.eval()
        self.generator.train_flag = False

    def _run_model(self, tensors):
        if self.gdEncoder is None or self.generator is None:
            raise RuntimeError("STDAN velocity model is not loaded")
        hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls = tensors
        lat_enc = torch.zeros(1, 3, device=self.device)
        lon_enc = torch.zeros(1, 3, device=self.device)
        self.gdEncoder.train() if self.mc_dropout else self.gdEncoder.eval()
        self.generator.eval()
        self.generator.train_flag = False
        with torch.no_grad():
            values = self.gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            fut_vel_modes, lat_pred, lon_pred = self.generator(values, lat_enc, lon_enc)
        self.gdEncoder.eval()
        return fut_vel_modes, lat_pred, lon_pred, mask

    def _decode_raw_prediction(
        self,
        target_id: int,
        target_state: Sequence[float],
        model_output,
        model_yaw: Optional[float] = None,
    ):
        fut_vel_modes, lat_pred, lon_pred, mask = model_output
        if len(fut_vel_modes) != self.args["lat_length"] * self.args["lon_length"]:
            raise ValueError(f"Expected 9 STDAN velocity modes, got {len(fut_vel_modes)}")

        vel_distributions = []
        for mode in fut_vel_modes:
            arr = mode.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = np.squeeze(arr, axis=1)
            vel_distributions.append(arr)
        raw_vel_dist_9 = np.asarray(vel_distributions, dtype=np.float64)
        raw_vel_xy_9 = raw_vel_dist_9[:, :, 0:2] * FT2M

        lat_probs = _normalized_prob(lat_pred, self.args["lat_length"])
        lon_probs = _normalized_prob(lon_pred, self.args["lon_length"])

        raw_vel_xy = np.zeros(
            (self.args["lat_length"], raw_vel_xy_9.shape[1], raw_vel_xy_9.shape[2]),
            dtype=np.float64,
        )
        raw_vel_dist = np.zeros(
            (self.args["lat_length"], raw_vel_dist_9.shape[1], raw_vel_dist_9.shape[2]),
            dtype=np.float64,
        )
        for lat_idx in range(self.args["lat_length"]):
            mode_idx = self.longitudinal_mode_idx * self.args["lat_length"] + lat_idx
            raw_vel_xy[lat_idx] = raw_vel_xy_9[mode_idx]
            raw_vel_dist[lat_idx] = raw_vel_dist_9[mode_idx]

        disp_ngsim = np.cumsum(raw_vel_xy * self.dt, axis=1)
        disp_rhs = np.stack((disp_ngsim[:, :, 1], -disp_ngsim[:, :, 0]), axis=-1)
        target_state = self._target_frame_state(target_state, model_yaw)
        raw_traj_xy = transform_points(
            disp_rhs.reshape(-1, 2),
            target_state[0],
            target_state[1],
            target_state[2],
            inverse=True,
        ).reshape(raw_vel_xy.shape[0], raw_vel_xy.shape[1], 2)
        covariance_moments = self._distribution_covariances_global(raw_vel_dist, target_state)

        prediction = {
            "vehicle_id": int(target_id),
            "raw_traj_xy": raw_traj_xy,
            "raw_pred_vel": raw_vel_xy,
            "raw_pred_vel_distribution": raw_vel_dist,
            "raw_pred_vel_distribution_9mode": raw_vel_dist_9,
            "raw_pred_vel_9mode": raw_vel_xy_9,
            "raw_intention_prob": lat_probs,
            "raw_longitudinal_prob": lon_probs,
            "selected_longitudinal_idx": self.longitudinal_mode_idx,
            "selected_longitudinal_mode": self.longitudinal_mode_names[self.longitudinal_mode_idx],
            "raw_intention_logits": None,
            "signed_t_cross": 0.0,
            "valid_mask": np.ones(raw_traj_xy.shape[:2], dtype=bool),
            "mask_true_count": int(mask.sum().item()),
            "model_yaw": None if model_yaw is None else float(model_yaw),
        }
        if covariance_moments is not None:
            velocity_cov_global, position_cov_global = covariance_moments
            prediction["raw_velocity_cov_global"] = velocity_cov_global
            prediction["raw_position_cov_global"] = position_cov_global
        return prediction


def resolve_default_checkpoint() -> str:
    return str(DEFAULT_CKPT)
