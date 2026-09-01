"""
Synthetic two-mode cut-out prediction provider for ACC Nair SMPC validation.

The target starts as the ego-lane lead vehicle.  The lane_keeping mode keeps it
blocking in the ego lane, while the cutout mode moves it to the adjacent lane
and makes it non-blocking after enough lateral clearance is predicted.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from utils.acc_nair_smpc import MultimodalLeadPrediction


@dataclass
class SyntheticCutOutConfig:
    horizon: int = 10
    dt: float = 0.2
    probabilities: Sequence[float] = (0.4, 0.6)
    lane_keeping_accel: float = 0.0
    cutout_accel: float = 0.0
    ego_lane_d: float = 0.0
    adjacent_lane_d: float = 3.5
    cutout_duration: float = 1.5
    cutout_start_delay: float = 0.5
    nonblocking_d_threshold: float = 1.75
    nonblocking_gap: float = 1000.0
    v_min: float = 0.0
    longitudinal_position_variance: float = 0.25
    velocity_variance: float = 0.09
    lateral_variance: float = 0.16
    variance_growth: float = 0.08


@dataclass
class SyntheticCutOutPrediction:
    mode_names: List[str]
    probabilities: np.ndarray
    prediction: MultimodalLeadPrediction
    frenet_mean: np.ndarray
    covariance_2d: np.ndarray


class SyntheticLaneKeepingCutOutPredictor:
    """Procedural lane-keeping/cut-out prediction generator for validation."""

    mode_names = ["lane_keeping", "cutout"]

    def __init__(self, config: Optional[SyntheticCutOutConfig] = None):
        self.config = config if config is not None else SyntheticCutOutConfig()

    def predict(self, ego_state, target_state, probabilities=None):
        """
        Build a two-mode cut-out prediction.

        :param ego_state: [s_ego, v_ego]
        :param target_state: [s_tv, d_tv, v_tv]
        :param probabilities: optional [p_lane_keeping, p_cutout]
        """
        cfg = self.config
        probs = np.asarray(probabilities if probabilities is not None else cfg.probabilities, dtype=float)
        probs = probs / np.sum(probs)

        ego_state = np.asarray(ego_state, dtype=float)
        target_state = np.asarray(target_state, dtype=float)
        lane_keeping = self._rollout_mode(target_state, cfg.lane_keeping_accel, cutout=False)
        cutout = self._rollout_mode(target_state, cfg.cutout_accel, cutout=True)

        effective = np.zeros((2, cfg.horizon + 1, 2))
        current_target_nonblocking = (
            abs(target_state[1] - cfg.ego_lane_d) >= cfg.nonblocking_d_threshold
        )
        lane_keeping_s = lane_keeping[:, 0]
        if current_target_nonblocking:
            # After the lead has actually cut out, lane_keeping means it keeps
            # the adjacent lane, not that it remains an ego-lane lead.
            lane_keeping_s = np.maximum(lane_keeping_s, ego_state[0] + cfg.nonblocking_gap)
        effective[0, :, 0] = lane_keeping_s
        effective[0, :, 1] = lane_keeping[:, 2]

        lateral_clear = np.abs(cutout[:, 1] - cfg.ego_lane_d) >= cfg.nonblocking_d_threshold
        nonblocking_s = np.maximum(cutout[:, 0], ego_state[0] + cfg.nonblocking_gap)
        effective[1, :, 0] = np.where(lateral_clear, nonblocking_s, cutout[:, 0])
        effective[1, :, 1] = cutout[:, 2]

        prediction = MultimodalLeadPrediction(
            means=effective,
            probabilities=probs,
            covariances=self._controller_covariances(),
            mode_names=list(self.mode_names),
        )
        frenet_mean = np.stack((lane_keeping, cutout), axis=0)
        return SyntheticCutOutPrediction(
            mode_names=list(self.mode_names),
            probabilities=probs,
            prediction=prediction,
            frenet_mean=frenet_mean,
            covariance_2d=self._lateral_covariances(),
        )

    def _rollout_mode(self, target_state, acceleration, cutout):
        cfg = self.config
        states = np.zeros((cfg.horizon + 1, 3), dtype=float)
        states[0] = target_state
        alpha = self._cutout_alpha()
        d_start = target_state[1]
        for step in range(cfg.horizon):
            states[step + 1, 2] = max(cfg.v_min, states[step, 2] + cfg.dt * acceleration)
            states[step + 1, 0] = states[step, 0] + cfg.dt * states[step, 2]
        if cutout:
            states[:, 1] = d_start + alpha * (cfg.adjacent_lane_d - d_start)
        else:
            states[:, 1] = d_start
        return states

    def _cutout_alpha(self):
        cfg = self.config
        times = np.arange(cfg.horizon + 1, dtype=float) * cfg.dt
        raw = np.clip((times - cfg.cutout_start_delay) / max(cfg.cutout_duration, cfg.dt), 0.0, 1.0)
        return raw * raw * (3.0 - 2.0 * raw)

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
