from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from predictor.stdan.stdan_predictor import M2FT, FT2M, transform_points
from predictor.stdan_3int_signed_tcross_velint.configs.config_3int_signed_tcross import args as BASE_ARGS

from .acc_postprocess import (
    REL_OTHER,
    build_multitarget_lead_prediction,
    global_to_frenet_trajectory,
    process_vehicle_prediction,
)


DEFAULT_CKPT = (
    Path(__file__).resolve().parent
    / "checkpoints_out30"
    / "best_model.pt"
)


class STDAN3IntACCAdapter:
    """
    CARLA-facing adapter from the 3-intention STDAN velocity model to 1D ACC.

    The model input contract is copied from the existing CARLA STDAN wrapper:
    target-centered NGSIM axes [lateral, longitudinal], meters converted to
    feet before inference, 31 history samples plus two extra samples to derive
    velocity and signed acceleration.
    """

    mode_names = ["LK", "LLC", "RLC"]
    _INV_STD_EPS = 1.0e-6

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        history: float = 3.0,
        future: float = 3.0,
        dt: float = 0.1,
        device: Optional[str] = None,
        mc_dropout: bool = False,
        load_model: bool = True,
        pad_short_history: bool = True,
        min_history_samples: int = 3,
    ):
        self.args = dict(BASE_ARGS)
        self.dt = float(dt)
        self.history = float(history)
        self.future = float(future)
        self.args["dt"] = self.dt
        self.args["in_length"] = int(round(history / dt)) + 1
        self.args["out_length"] = int(round(future / dt))
        self.args["num_intentions"] = 3
        self.args["train_flag"] = False
        if device is not None:
            self.args["device"] = torch.device(device)
        self.device = self.args["device"]
        self.in_length = int(self.args["in_length"])
        self.out_length = int(self.args["out_length"])
        self.full_length = self.in_length + 2
        self.encoder_size = int(self.args["lstm_encoder_size"])
        self.grid_size = (13, 3)
        self.mc_dropout = bool(mc_dropout)
        self.pad_short_history = bool(pad_short_history)
        self.min_history_samples = int(min_history_samples)
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
        if not self.ckpt_path.is_absolute():
            root = Path(__file__).resolve().parents[2]
            self.ckpt_path = root / self.ckpt_path
        self.gdEncoder = None
        self.generator = None
        if load_model:
            self._load_model(self.ckpt_path)

    def _load_model(self, ckpt_path: Path) -> None:
        from predictor.stdan_3int_signed_tcross_velint.models.model5f_3int_signed_tcross import (
            GDEncoder,
            Generator,
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"STDAN 3-int checkpoint not found: {ckpt_path}")
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

    def _empty_neighbor_tensors(self, n_objs: int = 0):
        return (
            torch.zeros(self.in_length, n_objs, 2, device=self.device),
            torch.zeros(self.in_length, n_objs, 2, device=self.device),
            torch.zeros(self.in_length, n_objs, 1, device=self.device),
            torch.full((self.in_length, n_objs, 1), 2.0, device=self.device),
        )

    def _base_tensors(self):
        hist = torch.zeros(self.in_length, 1, 2, device=self.device)
        va = torch.zeros(self.in_length, 1, 2, device=self.device)
        mask = torch.zeros(1, self.grid_size[1], self.grid_size[0], self.encoder_size, device=self.device)
        lane = torch.zeros(self.in_length, 1, 1, device=self.device)
        cls = torch.full((self.in_length, 1, 1), 2.0, device=self.device)
        return hist, va, mask, lane, cls

    def _grid_edges(self):
        num_lon_cells, num_lat_cells = self.grid_size
        cell_lat_size, cell_lon_size = 3.5, 4.6
        lat_min = -(num_lat_cells // 2) * cell_lat_size - cell_lat_size / 2
        lat_max = (num_lat_cells // 2) * cell_lat_size + cell_lat_size / 2
        lon_min = -(num_lon_cells // 2) * cell_lon_size - cell_lon_size / 2
        lon_max = (num_lon_cells // 2) * cell_lon_size + cell_lon_size / 2
        return (
            np.linspace(lat_min, lat_max, num_lat_cells + 1),
            np.linspace(lon_min, lon_max, num_lon_cells + 1),
        )

    def _has_enough_history(self, history: np.ndarray) -> bool:
        required = self.min_history_samples if self.pad_short_history else self.full_length
        return len(history) >= required

    @staticmethod
    def _target_frame_state(target_state: Sequence[float], model_yaw: Optional[float] = None) -> np.ndarray:
        frame_state = np.asarray(target_state, dtype=np.float64).copy()
        if model_yaw is not None:
            frame_state[2] = float(model_yaw)
        return frame_state

    @staticmethod
    def _rotation_matrix(yaw: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(float(yaw)), -np.sin(float(yaw))],
                [np.sin(float(yaw)), np.cos(float(yaw))],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _transform_covariances(covariances: np.ndarray, transform: np.ndarray) -> np.ndarray:
        transformed = np.einsum("ab,...bc,dc->...ad", transform, covariances, transform)
        return 0.5 * (transformed + np.swapaxes(transformed, -1, -2))

    def _distribution_covariances_global(
        self,
        raw_vel_dist: np.ndarray,
        target_state: Sequence[float],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        raw_vel_dist = np.asarray(raw_vel_dist, dtype=np.float64)
        if raw_vel_dist.ndim != 3 or raw_vel_dist.shape[-1] < 5:
            return None

        inv_sig_v = np.maximum(raw_vel_dist[:, :, 2:4], self._INV_STD_EPS)
        sigma_v = FT2M / inv_sig_v
        rho_v = np.clip(raw_vel_dist[:, :, 4], -0.99, 0.99)

        cov_v_ngsim = np.zeros(raw_vel_dist.shape[:2] + (2, 2), dtype=np.float64)
        cov_v_ngsim[:, :, 0, 0] = sigma_v[:, :, 0] ** 2
        cov_v_ngsim[:, :, 1, 1] = sigma_v[:, :, 1] ** 2
        cov_xy = rho_v * sigma_v[:, :, 0] * sigma_v[:, :, 1]
        cov_v_ngsim[:, :, 0, 1] = cov_xy
        cov_v_ngsim[:, :, 1, 0] = cov_xy

        cov_p_ngsim = np.cumsum(cov_v_ngsim * (self.dt ** 2), axis=1)

        ngsim_to_rhs = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        rhs_to_global = self._rotation_matrix(float(np.asarray(target_state, dtype=np.float64)[2]))
        ngsim_to_global = rhs_to_global @ ngsim_to_rhs
        velocity_cov_global = self._transform_covariances(cov_v_ngsim, ngsim_to_global)
        position_cov_global = self._transform_covariances(cov_p_ngsim, ngsim_to_global)
        return velocity_cov_global, position_cov_global

    @staticmethod
    def _global_covariances_to_frenet(
        covariances_xy: np.ndarray,
        points_xy: np.ndarray,
        frenet_handler,
    ) -> np.ndarray:
        covariances_xy = np.asarray(covariances_xy, dtype=np.float64)
        points_xy = np.asarray(points_xy, dtype=np.float64)
        if covariances_xy.shape[:2] != points_xy.shape[:2] or covariances_xy.shape[-2:] != (2, 2):
            raise ValueError(
                "covariances_xy must have shape (num_modes, steps, 2, 2) "
                "matching points_xy (num_modes, steps, 2)"
            )

        trajectory = np.asarray(frenet_handler.trajectory, dtype=np.float64)
        xy_traj = trajectory[:, 1:3]
        yaw_traj = trajectory[:, 3]
        flat_points = points_xy.reshape(-1, 2)
        flat_covariances = covariances_xy.reshape(-1, 2, 2)
        flat_output = np.zeros_like(flat_covariances)
        for idx, (point, covariance) in enumerate(zip(flat_points, flat_covariances)):
            closest_index = int(np.argmin(np.linalg.norm(xy_traj - point.reshape(1, 2), axis=1)))
            psi = float(yaw_traj[closest_index])
            rot_global_to_frenet = np.array(
                [
                    [np.cos(psi), np.sin(psi)],
                    [-np.sin(psi), np.cos(psi)],
                ],
                dtype=np.float64,
            )
            flat_output[idx] = rot_global_to_frenet @ covariance @ rot_global_to_frenet.T
        output = flat_output.reshape(covariances_xy.shape)
        return 0.5 * (output + np.swapaxes(output, -1, -2))

    @staticmethod
    def _lead_covariance_from_frenet_moments(
        position_cov_frenet: np.ndarray,
        velocity_cov_frenet: np.ndarray,
    ) -> np.ndarray:
        position_cov_frenet = np.asarray(position_cov_frenet, dtype=np.float64)
        velocity_cov_frenet = np.asarray(velocity_cov_frenet, dtype=np.float64)
        if position_cov_frenet.shape != velocity_cov_frenet.shape:
            raise ValueError("position and velocity covariance shapes must match")
        lead_covariance = np.zeros_like(position_cov_frenet)
        lead_covariance[:, :, 0, 0] = np.maximum(position_cov_frenet[:, :, 0, 0], 0.0)
        lead_covariance[:, :, 1, 1] = np.maximum(velocity_cov_frenet[:, :, 0, 0], 0.0)
        return lead_covariance

    def _history_to_ngsim(
        self,
        history: np.ndarray,
        target_state: Sequence[float],
        model_yaw: Optional[float] = None,
    ) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)[-self.full_length :, :2]
        if history.shape[0] < self.full_length:
            if not self.pad_short_history or history.shape[0] < self.min_history_samples:
                raise ValueError(
                    f"history length {history.shape[0]} is too short for STDAN input"
                )
            pad = np.repeat(history[:1], self.full_length - history.shape[0], axis=0)
            history = np.vstack((pad, history))
        frame_state = self._target_frame_state(target_state, model_yaw)
        local_rhs = transform_points(history[:, :2], frame_state[0], frame_state[1], frame_state[2])
        return np.column_stack((-local_rhs[:, 1], local_rhs[:, 0]))

    def _motion_tensors_from_ngsim(self, ngsim_history_m: np.ndarray):
        history_ft = torch.from_numpy(ngsim_history_m * M2FT).float().to(self.device)
        velocity = (history_ft[1:] - history_ft[:-1]) / self.dt
        acceleration = (velocity[1:] - velocity[:-1]) / self.dt
        hist = history_ft[-self.in_length :]
        final_velocity = velocity[-self.in_length :]
        speed = torch.norm(final_velocity, p=2, dim=-1, keepdim=True)
        signed_acceleration = (acceleration * final_velocity).sum(dim=-1, keepdim=True) / (speed + 1e-8)
        return hist, torch.cat((speed, signed_acceleration), dim=-1)

    def make_tensor(
        self,
        target_id: int,
        target_state: Sequence[float],
        trackings: Dict[int, np.ndarray],
        model_yaw: Optional[float] = None,
    ):
        hist_tsr, va_tsr, mask_tsr, lane_tsr, cls_tsr = self._base_tensors()
        if target_id not in trackings or not self._has_enough_history(trackings[target_id]):
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = self._empty_neighbor_tensors()
            return (
                hist_tsr,
                nbrs_tsr,
                mask_tsr.bool(),
                va_tsr,
                nbrsva_tsr,
                lane_tsr,
                nbrslane_tsr,
                cls_tsr,
                nbrscls_tsr,
            )

        target_ngsim = self._history_to_ngsim(trackings[target_id], target_state, model_yaw=model_yaw)
        hist_tsr[:, 0, :], va_tsr[:, 0, :] = self._motion_tensors_from_ngsim(target_ngsim)

        lat_edges, lon_edges = self._grid_edges()
        valid_neighbors = []
        for nbr_id, history in trackings.items():
            if nbr_id == target_id or not self._has_enough_history(history):
                continue
            nbr_ngsim = self._history_to_ngsim(history, target_state, model_yaw=model_yaw)
            nbr_hist, nbr_va = self._motion_tensors_from_ngsim(nbr_ngsim)
            nbr_current = nbr_ngsim[-1]
            grid_lat = np.digitize(nbr_current[0], lat_edges) - 1
            grid_lon = np.digitize(nbr_current[1], lon_edges) - 1
            if 0 <= grid_lat < self.grid_size[1] and 0 <= grid_lon < self.grid_size[0]:
                flat_grid_index = grid_lat * self.grid_size[0] + grid_lon
                valid_neighbors.append((flat_grid_index, nbr_hist, nbr_va))
                mask_tsr[0, grid_lat, grid_lon, :] = 1.0

        if valid_neighbors:
            valid_neighbors.sort(key=lambda item: item[0])
            nbrs_tsr = torch.stack([item[1] for item in valid_neighbors], dim=1)
            nbrsva_tsr = torch.stack([item[2] for item in valid_neighbors], dim=1)
            n_objs = len(valid_neighbors)
            nbrslane_tsr = torch.zeros(self.in_length, n_objs, 1, device=self.device)
            nbrscls_tsr = torch.full((self.in_length, n_objs, 1), 2.0, device=self.device)
        else:
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = self._empty_neighbor_tensors()

        return (
            hist_tsr,
            nbrs_tsr,
            mask_tsr.bool(),
            va_tsr,
            nbrsva_tsr,
            lane_tsr,
            nbrslane_tsr,
            cls_tsr,
            nbrscls_tsr,
        )

    def _run_model(self, tensors):
        if self.gdEncoder is None or self.generator is None:
            raise RuntimeError("STDAN 3-int model is not loaded")
        hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls = tensors
        intent_enc = torch.zeros(1, 3, device=self.device)
        self.gdEncoder.train() if self.mc_dropout else self.gdEncoder.eval()
        self.generator.eval()
        self.generator.train_flag = False
        with torch.no_grad():
            values = self.gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            fut_vel_modes, intent_prob, t_cross_pred = self.generator(values, intent_enc)
        self.gdEncoder.eval()
        return fut_vel_modes, intent_prob, t_cross_pred, mask

    def _decode_raw_prediction(
        self,
        target_id: int,
        target_state: Sequence[float],
        model_output,
        model_yaw: Optional[float] = None,
    ):
        fut_vel_modes, intent_prob, t_cross_pred, mask = model_output
        velocities = []
        for mode in fut_vel_modes:
            arr = mode.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = np.squeeze(arr, axis=1)
            velocities.append(arr)
        raw_vel_dist = np.asarray(velocities, dtype=np.float64)
        raw_vel_xy = raw_vel_dist[:, :, 0:2] * FT2M
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
        probs = intent_prob.detach().cpu().numpy().reshape(-1)
        probs = probs / max(float(np.sum(probs)), 1.0e-12)
        prediction = {
            "vehicle_id": int(target_id),
            "raw_traj_xy": raw_traj_xy,
            "raw_pred_vel": raw_vel_xy,
            "raw_pred_vel_distribution": raw_vel_dist,
            "raw_intention_prob": probs,
            "raw_intention_logits": None,
            "signed_t_cross": float(t_cross_pred.detach().cpu().numpy().reshape(-1)[0] * 28.0),
            "valid_mask": np.ones(raw_traj_xy.shape[:2], dtype=bool),
            "mask_true_count": int(mask.sum().item()),
            "model_yaw": None if model_yaw is None else float(model_yaw),
        }
        if covariance_moments is not None:
            velocity_cov_global, position_cov_global = covariance_moments
            prediction["raw_velocity_cov_global"] = velocity_cov_global
            prediction["raw_position_cov_global"] = position_cov_global
        return prediction

    def predict_raw(
        self,
        target_ids: Iterable[int],
        trackings: Dict[int, np.ndarray],
        model_yaw: Optional[float] = None,
    ) -> Dict[int, dict]:
        predictions = {}
        for target_id in target_ids:
            if target_id not in trackings or not self._has_enough_history(trackings[target_id]):
                continue
            target_state = np.asarray(trackings[target_id], dtype=np.float64)[-1]
            tensors = self.make_tensor(int(target_id), target_state, trackings, model_yaw=model_yaw)
            model_output = self._run_model(tensors)
            predictions[int(target_id)] = self._decode_raw_prediction(
                int(target_id),
                target_state,
                model_output,
                model_yaw=model_yaw,
            )
        return predictions

    def predict_acc(
        self,
        ego_state: Sequence[float],
        target_states_frenet: Dict[int, Sequence[float]],
        target_relations: Dict[int, str],
        trackings: Dict[int, np.ndarray],
        frenet_handler,
        horizon: int,
        desired_speed: float,
        num_modes: int,
        ego_lane_d: float = 0.0,
        ego_lane_threshold: float = 0.5,
        controller_dt: Optional[float] = None,
        lane_membership_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
        model_yaw: Optional[float] = None,
        cutin_probability_threshold: float = 0.0,
        cutin_clearance_ramp_ref: float = 0.0,
        cutin_clearance_tlc_ref: float = 0.0,
        gap_recovery_elapsed: Optional[Dict[int, float]] = None,
        gap_recovery_s: float = 0.0,
        gap_recovery_floor: float = 0.6,
        gap_recovery_start_scale: Optional[Dict[int, float]] = None,
    ):
        controller_dt = float(self.dt if controller_dt is None else controller_dt)
        raw_predictions = self.predict_raw(target_states_frenet.keys(), trackings, model_yaw=model_yaw)
        processed = []
        for target_id, raw in raw_predictions.items():
            traj_frenet = []
            for mode_xy in raw["raw_traj_xy"]:
                traj_frenet.append(global_to_frenet_trajectory(mode_xy, frenet_handler))
            raw["full_pred_traj_frenet"] = np.asarray(traj_frenet, dtype=float)
            raw["full_raw_pred_vel"] = np.asarray(raw["raw_pred_vel"], dtype=float)
            if "raw_position_cov_global" in raw and "raw_velocity_cov_global" in raw:
                position_cov_frenet = self._global_covariances_to_frenet(
                    raw["raw_position_cov_global"],
                    raw["raw_traj_xy"],
                    frenet_handler,
                )
                velocity_cov_frenet = self._global_covariances_to_frenet(
                    raw["raw_velocity_cov_global"],
                    raw["raw_traj_xy"],
                    frenet_handler,
                )
                full_lead_covariance = self._lead_covariance_from_frenet_moments(
                    position_cov_frenet,
                    velocity_cov_frenet,
                )
                raw["full_pred_position_cov_frenet"] = position_cov_frenet
                raw["full_pred_velocity_cov_frenet"] = velocity_cov_frenet
                raw["full_pred_lead_covariance"] = full_lead_covariance
            raw["pred_traj_frenet"] = self._sample_future(np.asarray(traj_frenet, dtype=float), horizon, controller_dt)
            raw["raw_pred_vel"] = self._sample_future(np.asarray(raw["raw_pred_vel"], dtype=float), horizon, controller_dt)
            if "full_pred_lead_covariance" in raw:
                raw["pred_lead_covariance"] = self._sample_future(
                    np.asarray(raw["full_pred_lead_covariance"], dtype=float),
                    horizon,
                    controller_dt,
                )
            raw["sampled_raw_traj_xy"] = self._sample_future(
                np.asarray(raw["raw_traj_xy"], dtype=float),
                horizon,
                controller_dt,
            )
            mode_lane_memberships = None
            lane_membership_source = "frenet_d_threshold"
            if lane_membership_fn is not None:
                mode_lane_memberships = lane_membership_fn(int(target_id), raw["sampled_raw_traj_xy"])
                lane_membership_source = "carla_waypoint"
            processed.append(
                process_vehicle_prediction(
                    raw,
                    target_relations.get(target_id, REL_OTHER),
                    target_states_frenet[target_id],
                    horizon=horizon,
                    dt=controller_dt,
                    ego_lane_d=ego_lane_d,
                    ego_lane_threshold=ego_lane_threshold,
                    mode_lane_memberships=mode_lane_memberships,
                    lane_membership_source=lane_membership_source,
                    cutin_probability_threshold=cutin_probability_threshold,
                    cutin_clearance_ramp_ref=cutin_clearance_ramp_ref,
                    cutin_clearance_tlc_ref=cutin_clearance_tlc_ref,
                    gap_recovery_elapsed_s=(
                        None if gap_recovery_elapsed is None
                        else gap_recovery_elapsed.get(target_id)),
                    gap_recovery_s=gap_recovery_s,
                    gap_recovery_floor=gap_recovery_floor,
                    gap_recovery_start_scale=(
                        None if gap_recovery_start_scale is None
                        else gap_recovery_start_scale.get(target_id)),
                )
            )
        prediction, metadata = build_multitarget_lead_prediction(
            processed,
            ego_state=ego_state,
            horizon=horizon,
            desired_speed=desired_speed,
            num_modes=num_modes,
        )
        return {
            "prediction": prediction,
            "processed_targets": processed,
            "scenario_metadata": metadata,
            "raw_predictions": raw_predictions,
        }

    def _sample_future(self, values: np.ndarray, horizon: int, controller_dt: float) -> np.ndarray:
        values = np.asarray(values)
        max_idx = values.shape[1] - 1
        indices = []
        for step in range(1, int(horizon) + 1):
            model_idx = int(round((step * controller_dt) / self.dt)) - 1
            indices.append(int(np.clip(model_idx, 0, max_idx)))
        return values[:, indices, ...]


def resolve_default_checkpoint() -> str:
    return os.fspath(DEFAULT_CKPT)
