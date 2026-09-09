from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from predictor.lstm.acc_adapter import LSTMACCAdapter
from predictor.stdan.stdan_predictor import FT2M


DEFAULT_CKPT = Path(__file__).resolve().parent / "ckpt" / "best_model.pt"

#: Widest gain the speed anchor may apply.  A prediction that needs more than
#: this is not mis-scaled, it is wrong, and stretching it would invent motion.
GAIN_LIMITS = (0.5, 2.0)
#: Samples of history averaged for the measured speed.  One step is the
#: tracker's own quantisation; three is still well inside a lane change.
SPEED_SAMPLES = 3
#: Predicted steps averaged for the anchor.  ``None`` means the whole horizon.
#: Reading the gain off the first 0.5 s instead was tried and reverted
#: (2026-09-09, 96 runs): it was meant to keep a predicted deceleration from
#: being stretched away, and on a synthetic 2 m/s^2 decel it does, but in CARLA
#: it changed nothing in the cut-in cells it was aimed at (delta_max 50.16 ->
#: 50.35, dt_ant 0.28 -> 0.27) and cost the cut-out cells their gain, because a
#: cut-out lead is mostly constant speed and five steps carry the decoder's
#: noise into the gain: excess gap error 3.13 -> 10.55 (05), 11.03 -> 17.51
#: (06), 15.87 -> 21.15 (07).  The cut-in regression it was chasing is more
#: likely the jerk limit -- this policy saturates it in all 36 cut-in cells --
#: than the anchor.
ANCHOR_STEPS = None


class LSTMVelACCAdapter(LSTMACCAdapter):
    """The ``lstm_vel`` checkpoint predicts velocity, and its scale is anchored.

    Same architecture as ``predictor/lstm`` -- ``model5f_lstm.py`` is byte
    identical -- trained against a velocity target, so the decoder emits ft/s
    per step where the position model emits ft.  Integrating it is the whole
    difference; everything downstream (``smooth_constant_acceleration``, the
    NGSIM-to-world transform, the finite-difference speed) is inherited and
    sees displacement in ft either way.

    On top of that the horizon is rescaled so its mean speed matches the
    target's *measured* speed.  Both LSTM checkpoints read a constant-speed
    straight line about 10 % slow, and the error grows with speed (measured
    2026-09-09 against a synthetic straight history: -6 % at 7 m/s, -12 % at
    13, -17 % at 17 for the position model; -3 %/-10 %/-14 % for this one),
    while STDAN on the same input is within +2 %.  That is a calibration
    defect rather than a prediction one, and left alone it costs the LSTM
    baseline far more than its trajectory quality does -- it lags every lead
    and holds a gap error the controller cannot close.

    The anchor uses only the speed a tracker already measures now, never the
    future the model is being asked for, so it is not oracle information: any
    deployed stack has it.  It is also a strictly *stronger* baseline, which is
    the conservative direction for a paper comparing against it.  The gain is
    one scalar over both axes, so the predicted shape -- lane change included
    -- and the predicted acceleration ratio are untouched.
    """

    def __init__(self, ckpt_path: Optional[str] = None, **kwargs):
        super().__init__(ckpt_path=ckpt_path or str(DEFAULT_CKPT), **kwargs)

    def _run_model(self, tensors) -> np.ndarray:
        velocity_ft_s = np.asarray(super()._run_model(tensors), dtype=np.float64)
        return np.cumsum(velocity_ft_s * self.dt, axis=0)

    def _measured_speed(self, track: np.ndarray) -> Optional[float]:
        """Speed [m/s] from the last few tracked positions, or None."""
        xy = np.asarray(track, dtype=np.float64)[:, :2]
        if xy.shape[0] < 2:
            return None
        steps = min(int(SPEED_SAMPLES), xy.shape[0] - 1)
        travelled = float(np.linalg.norm(xy[-1] - xy[-1 - steps]))
        return travelled / (steps * self.dt)

    def _speed_anchor_gain(self, track: np.ndarray, disp_ft: np.ndarray) -> float:
        measured = self._measured_speed(track)
        if measured is None or measured <= 0.1:
            return 1.0
        steps = disp_ft.shape[0] if ANCHOR_STEPS is None else min(int(ANCHOR_STEPS), disp_ft.shape[0])
        predicted = float(np.linalg.norm(disp_ft[steps - 1])) * FT2M / (steps * self.dt)
        if predicted <= 0.1:
            return 1.0
        return float(np.clip(measured / predicted, *GAIN_LIMITS))

    def predict_raw(
        self,
        target_ids: Iterable[int],
        trackings: Dict[int, np.ndarray],
        model_yaw: Optional[float] = None,
    ) -> Dict[int, dict]:
        predictions = {}
        for target_id in target_ids:
            track = trackings.get(target_id)
            if track is None or not self._stdan_io._has_enough_history(track):
                continue
            target_state = np.asarray(track, dtype=np.float64)[-1]
            tensors = self.make_tensor(int(target_id), target_state, trackings, model_yaw=model_yaw)
            disp_ft = self._run_model(tensors)
            disp_ft = disp_ft * self._speed_anchor_gain(track, disp_ft)
            predictions[int(target_id)] = self._decode_raw_prediction(
                int(target_id), target_state, disp_ft, model_yaw=model_yaw,
            )
        return predictions
