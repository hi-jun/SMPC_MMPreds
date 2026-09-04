from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from predictor.stdan.stdan_predictor import FT2M, transform_points
from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (
    ACCModePrediction,
    ACCProcessedPrediction,
    REL_OTHER,
    build_multitarget_lead_prediction,
    ego_lane_interval,
    ego_lane_membership_split_step_from_mask,
    global_to_frenet_trajectory,
    lane_occupancy_from_d,
)


DEFAULT_CKPT = Path(__file__).resolve().parent / "ckpt" / "best_model.pt"
MODE_NAME = "lstm"
# Architecture of predictor/lstm/config_lstm.py.  Importing that module reseeds
# the global random/numpy/torch generators, so the values are copied here.
MODEL_ARGS = {
    "f_length": 5,
    "traj_linear_hidden": 64,
    "lstm_encoder_size": 128,
    "dropout": 0.2,
    "use_elu": True,
    "relu": 0.1,
}


class LSTMACCAdapter:
    """
    CARLA-facing adapter from the vanilla LSTM baseline to 1D ACC.

    One deterministic future per target with probability 1: the "SCC + LSTM"
    baseline.  The model was trained on the same NGSIM pipeline as STDAN
    (``loader2_hdf5.NgsimDataset``: history and future are ``local_x, local_y``
    in feet relative to the vehicle's own position at the current frame, ``va``
    is NGSIM ``v_Vel``/``v_Acc``, class code 2 = car), so the input side reuses
    the STDAN adapter's conversions unchanged.  Unlike STDAN the LSTM sees only
    the target's own history (no social grid) and outputs positions, not
    velocities: ``getFuture`` returns ``vehTrack[t+1 : t+50, 1:3] - refPos``.
    """

    mode_names = [MODE_NAME]

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        history: float = 3.0,
        future: float = 5.0,
        dt: float = 0.1,
        device: str = "cpu",
        load_model: bool = True,
        pad_short_history: bool = True,
        min_history_samples: int = 3,
    ):
        # Owns the model-input contract (history window, NGSIM axes, feet,
        # speed and signed acceleration) so both predictors see identical tensors.
        self._stdan_io = STDAN3IntACCAdapter(
            history=history,
            future=future,
            dt=dt,
            device=device,
            load_model=False,
            pad_short_history=pad_short_history,
            min_history_samples=min_history_samples,
        )
        self.dt = self._stdan_io.dt
        self.history = self._stdan_io.history
        self.future = self._stdan_io.future
        self.in_length = self._stdan_io.in_length
        self.out_length = self._stdan_io.out_length
        self.full_length = self._stdan_io.full_length
        self.pad_short_history = self._stdan_io.pad_short_history
        self.min_history_samples = self._stdan_io.min_history_samples
        self.device = self._stdan_io.device
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else DEFAULT_CKPT
        if not self.ckpt_path.is_absolute():
            root = Path(__file__).resolve().parents[2]
            self.ckpt_path = root / self.ckpt_path
        self.model = None
        if load_model:
            self._load_model(self.ckpt_path)

    def _load_model(self, ckpt_path: Path) -> None:
        from predictor.lstm.model5f_lstm import VanillaLSTM

        if not ckpt_path.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
        args = dict(MODEL_ARGS, out_length=self.out_length)
        self.model = VanillaLSTM(args).to(self.device)
        self.model.load_state_dict(torch.load(str(ckpt_path), map_location=self.device))
        self.model.eval()

    def make_tensor(
        self,
        target_id: int,
        target_state: Sequence[float],
        trackings: Dict[int, np.ndarray],
        model_yaw: Optional[float] = None,
    ):
        ngsim = self._stdan_io._history_to_ngsim(trackings[target_id], target_state, model_yaw=model_yaw)
        hist, va = self._stdan_io._motion_tensors_from_ngsim(ngsim)
        cls = torch.full((self.in_length, 1, 1), 2.0, device=self.device)
        return hist[:, None, :], va[:, None, :], cls

    def _run_model(self, tensors) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM model is not loaded")
        hist, va, cls = tensors
        with torch.no_grad():
            fut = self.model(hist, va, cls)
        return fut[:, 0, :].detach().cpu().numpy()

    def _decode_raw_prediction(
        self,
        target_id: int,
        target_state: Sequence[float],
        disp_ngsim_ft: np.ndarray,
        model_yaw: Optional[float] = None,
    ):
        disp_ngsim = np.asarray(disp_ngsim_ft, dtype=np.float64) * FT2M
        disp_rhs = np.column_stack((disp_ngsim[:, 1], -disp_ngsim[:, 0]))
        frame_state = STDAN3IntACCAdapter._target_frame_state(target_state, model_yaw)
        raw_traj_xy = transform_points(
            disp_rhs,
            frame_state[0],
            frame_state[1],
            frame_state[2],
            inverse=True,
        )[None]
        # Finite-difference velocity in the NGSIM frame, like STDAN's
        # raw_pred_vel; only its norm is used downstream.
        raw_vel = np.diff(np.vstack((np.zeros((1, 2)), disp_ngsim)), axis=0)[None] / self.dt
        return {
            "vehicle_id": int(target_id),
            "raw_traj_xy": raw_traj_xy,
            "raw_pred_vel": raw_vel,
            "raw_intention_prob": np.array([1.0]),
            "valid_mask": np.ones(raw_traj_xy.shape[:2], dtype=bool),
            "model_yaw": None if model_yaw is None else float(model_yaw),
        }

    def predict_raw(
        self,
        target_ids: Iterable[int],
        trackings: Dict[int, np.ndarray],
        model_yaw: Optional[float] = None,
    ) -> Dict[int, dict]:
        predictions = {}
        for target_id in target_ids:
            if target_id not in trackings or not self._stdan_io._has_enough_history(trackings[target_id]):
                continue
            target_state = np.asarray(trackings[target_id], dtype=np.float64)[-1]
            tensors = self.make_tensor(int(target_id), target_state, trackings, model_yaw=model_yaw)
            predictions[int(target_id)] = self._decode_raw_prediction(
                int(target_id),
                target_state,
                self._run_model(tensors),
                model_yaw=model_yaw,
            )
        return predictions

    @staticmethod
    def _process_target(
        raw: dict,
        relation_to_ego_lane: str,
        current_frenet_state: Sequence[float],
        horizon: int,
        ego_lane_d: float,
        ego_lane_threshold: float,
        lane_membership_fn: Optional[Callable[[int, np.ndarray], np.ndarray]],
    ) -> ACCProcessedPrediction:
        """Single-mode counterpart of ``process_vehicle_prediction``.

        The target is a lead from the first step its predicted position lies in
        the ego lane and constrains nothing otherwise; step 0 is the measured
        state, so a target already in the ego lane stays active from now.
        """
        vehicle_id = int(raw["vehicle_id"])
        steps = int(horizon) + 1
        frenet = np.zeros((steps, 3), dtype=float)
        frenet[0] = np.asarray(current_frenet_state, dtype=float)[:3]
        frenet[1:, 0:2] = np.asarray(raw["pred_traj_frenet"], dtype=float)[0, : steps - 1, :2]
        frenet[1:, 2] = np.linalg.norm(np.asarray(raw["raw_pred_vel"], dtype=float)[0, : steps - 1, :2], axis=1)
        if lane_membership_fn is not None:
            membership = np.asarray(lane_membership_fn(vehicle_id, raw["sampled_raw_traj_xy"]), dtype=bool)[0]
            lane_membership_source = "carla_waypoint"
        else:
            membership = lane_occupancy_from_d(frenet, ego_lane_d, ego_lane_threshold)
            lane_membership_source = "frenet_d_threshold"
        if membership.shape != (steps,):
            raise ValueError(f"lane membership must have shape ({steps},), got {membership.shape}")
        memberships = membership[None]
        split_step = ego_lane_membership_split_step_from_mask(memberships)
        start_idx, end_idx = ego_lane_interval(membership)
        mode = ACCModePrediction(
            vehicle_id=vehicle_id,
            mode_name=MODE_NAME,
            probability=1.0,
            frenet=frenet,
            active_mask=membership.copy(),
            raw_mode_indices=[0],
            ego_lane_membership_split_step=split_step,
        )
        return ACCProcessedPrediction(
            vehicle_id=vehicle_id,
            relation_to_ego_lane=relation_to_ego_lane,
            mode_predictions=[mode],
            raw_mode_prob={MODE_NAME: 1.0},
            acc_mode_prob={MODE_NAME: 1.0},
            branch_info={
                "ego_lane_membership_split_step": split_step,
                "branch_step": split_step,
                "mode_lane_membership": memberships,
                "ego_lane_occupancy_mask": membership,
                "ego_lane_start_idx": start_idx,
                "ego_lane_end_idx": end_idx,
                "lane_membership_source": lane_membership_source,
            },
            raw_prediction=raw,
        )

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
        **ignored,
    ):
        """Same call as ``STDAN3IntACCAdapter.predict_acc``.

        ``ignored`` takes the cut-in relaxation and gap-recovery keywords
        (``cutin_*``, ``gap_recovery_*``).  They act on ``cutin`` modes, which
        this single-mode predictor never produces, so nothing is gated or scaled.
        """
        controller_dt = float(self.dt if controller_dt is None else controller_dt)
        horizon = int(horizon)
        raw_predictions = self.predict_raw(target_states_frenet.keys(), trackings, model_yaw=model_yaw)
        processed = []
        for target_id, raw in raw_predictions.items():
            raw["full_pred_traj_frenet"] = np.asarray(
                [global_to_frenet_trajectory(raw["raw_traj_xy"][0], frenet_handler)], dtype=float)
            raw["full_raw_pred_vel"] = raw["raw_pred_vel"]
            raw["pred_traj_frenet"] = self._stdan_io._sample_future(raw["full_pred_traj_frenet"], horizon, controller_dt)
            raw["raw_pred_vel"] = self._stdan_io._sample_future(raw["full_raw_pred_vel"], horizon, controller_dt)
            raw["sampled_raw_traj_xy"] = self._stdan_io._sample_future(raw["raw_traj_xy"], horizon, controller_dt)
            processed.append(
                self._process_target(
                    raw,
                    target_relations.get(target_id, REL_OTHER),
                    target_states_frenet[target_id],
                    horizon,
                    ego_lane_d,
                    ego_lane_threshold,
                    lane_membership_fn,
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
