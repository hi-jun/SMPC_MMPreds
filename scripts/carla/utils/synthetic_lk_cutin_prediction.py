"""
Synthetic two-mode prediction provider for ACC Nair SMPC validation.

This is intentionally procedural and only meant for integration smoke tests.
It produces one adjacent-lane lane-keeping mode and one cut-in mode, while the
1D ACC controller consumes the effective longitudinal lead state [s, v].
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from utils.acc_nair_smpc import MultimodalLeadPrediction, PolicySharingTree


@dataclass
class SyntheticCutInConfig:
    horizon: int = 8
    dt: float = 0.2
    probabilities: Sequence[float] = (0.6, 0.4)
    lane_keeping_accel: float = 0.0
    cutin_accel: float = -0.5
    ego_lane_d: float = 0.0
    cutin_duration: float = 3.0
    cutin_start_delay: float = 0.5
    blocking_alpha: float = 0.3
    nonblocking_gap: float = 1000.0
    ego_lane_blocking_d_threshold: float = 0.5
    v_min: float = 0.0
    longitudinal_position_variance: float = 0.25
    velocity_variance: float = 0.09
    lateral_variance: float = 0.16
    variance_growth: float = 0.08
    use_prediction_tree: bool = False
    branch_s_threshold: float = 1.0
    branch_v_threshold: float = 0.5


@dataclass
class SyntheticCutInPrediction:
    mode_names: List[str]
    probabilities: np.ndarray
    prediction: MultimodalLeadPrediction
    frenet_mean: np.ndarray
    covariance_2d: np.ndarray


class SyntheticLaneKeepingCutInPredictor:
    """Procedural LK/cut-in prediction generator for validation only."""

    mode_names = ["lane_keeping", "cutin"]

    def __init__(self, config: Optional[SyntheticCutInConfig] = None):
        self.config = config if config is not None else SyntheticCutInConfig()

    def predict(self, ego_state, target_state, probabilities=None):
        """
        Build a two-mode prediction.

        :param ego_state: [s_ego, v_ego]
        :param target_state: [s_tv, d_tv, v_tv]
        :param probabilities: optional [p_lk, p_cutin]
        """
        cfg = self.config
        probs = np.asarray(probabilities if probabilities is not None else cfg.probabilities, dtype=float)
        probs = probs / np.sum(probs)

        ego_state = np.asarray(ego_state, dtype=float)
        target_state = np.asarray(target_state, dtype=float)
        actual_lk = self._rollout_mode(target_state, cfg.lane_keeping_accel, cutin=False)
        actual_cutin = self._rollout_mode(target_state, cfg.cutin_accel, cutin=True)

        effective = np.zeros((2, cfg.horizon + 1, 2))
        target_in_ego_lane = abs(target_state[1] - cfg.ego_lane_d) <= cfg.ego_lane_blocking_d_threshold
        if target_in_ego_lane:
            # Once the cut-in has been observed as complete, "lane_keeping"
            # means the target keeps the ego lane and remains the effective
            # longitudinal lead vehicle.  Before that, lane_keeping is the
            # adjacent-lane non-blocking mode.
            effective[0, :, 0] = actual_lk[:, 0]
        else:
            effective[0, :, 0] = np.maximum(actual_lk[:, 0], ego_state[0] + cfg.nonblocking_gap)
        effective[0, :, 1] = actual_lk[:, 2]

        alpha = self._cutin_alpha()
        cutin_s = actual_cutin[:, 0].copy()
        nonblocking_s = np.maximum(cutin_s, ego_state[0] + cfg.nonblocking_gap)
        predicted_in_ego_lane = (
            np.abs(actual_cutin[:, 1] - cfg.ego_lane_d) <= cfg.ego_lane_blocking_d_threshold
        )
        blocking = np.logical_or(alpha >= cfg.blocking_alpha, predicted_in_ego_lane)
        effective[1, :, 0] = np.where(blocking, cutin_s, nonblocking_s)
        effective[1, :, 1] = actual_cutin[:, 2]

        covariances = self._controller_covariances()
        frenet_mean = np.stack((actual_lk, actual_cutin), axis=0)
        policy_tree = self._policy_tree(effective) if cfg.use_prediction_tree else None
        prediction = MultimodalLeadPrediction(
            means=effective,
            probabilities=probs,
            covariances=covariances,
            mode_names=list(self.mode_names),
            policy_tree=policy_tree,
        )
        return SyntheticCutInPrediction(
            mode_names=list(self.mode_names),
            probabilities=probs,
            prediction=prediction,
            frenet_mean=frenet_mean,
            covariance_2d=self._lateral_covariances(),
        )

    def _rollout_mode(self, target_state, acceleration, cutin):
        cfg = self.config
        states = np.zeros((cfg.horizon + 1, 3), dtype=float)
        states[0] = target_state
        alpha = self._cutin_alpha()
        d_start = target_state[1]
        for step in range(cfg.horizon):
            states[step + 1, 2] = max(cfg.v_min, states[step, 2] + cfg.dt * acceleration)
            states[step + 1, 0] = states[step, 0] + cfg.dt * states[step, 2]
        if cutin:
            states[:, 1] = d_start + alpha * (cfg.ego_lane_d - d_start)
        else:
            states[:, 1] = d_start
        return states

    def _cutin_alpha(self):
        cfg = self.config
        times = np.arange(cfg.horizon + 1, dtype=float) * cfg.dt
        raw = np.clip((times - cfg.cutin_start_delay) / max(cfg.cutin_duration, cfg.dt), 0.0, 1.0)
        return raw * raw * (3.0 - 2.0 * raw)

    def _policy_tree(self, controller_mean):
        cfg = self.config
        split_step = cfg.horizon
        reference = controller_mean[0]
        for step in range(cfg.horizon):
            deltas = np.abs(controller_mean[:, step, :] - reference[step])
            if (
                    np.any(deltas[:, 0] >= cfg.branch_s_threshold)
                    or np.any(deltas[:, 1] >= cfg.branch_v_threshold)):
                split_step = step
                break
        return PolicySharingTree.single_split(
            num_modes=len(self.mode_names),
            horizon=cfg.horizon,
            split_step=split_step,
            mode_names=list(self.mode_names),
        )

    def _controller_covariances(self):
        cfg = self.config
        covariances = np.zeros((2, cfg.horizon + 1, 2, 2), dtype=float)
        for mode in range(2):
            for step in range(cfg.horizon + 1):
                growth = 1.0 + cfg.variance_growth * step
                covariances[mode, step] = np.diag([
                    cfg.longitudinal_position_variance * growth,
                    cfg.velocity_variance * growth,
                ])
        return covariances

    def _lateral_covariances(self):
        cfg = self.config
        covariances = np.zeros((2, cfg.horizon + 1, 3, 3), dtype=float)
        for mode in range(2):
            for step in range(cfg.horizon + 1):
                growth = 1.0 + cfg.variance_growth * step
                covariances[mode, step] = np.diag([
                    cfg.longitudinal_position_variance * growth,
                    cfg.lateral_variance * growth,
                    cfg.velocity_variance * growth,
                ])
        return covariances
