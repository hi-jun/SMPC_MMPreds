from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from predictor.lstm.acc_adapter import LSTMACCAdapter


DEFAULT_CKPT = Path(__file__).resolve().parent / "ckpt" / "best_model.pt"


class LSTMVelACCAdapter(LSTMACCAdapter):
    """The ``lstm_vel`` checkpoint predicts velocity, not displacement.

    Same architecture as ``predictor/lstm`` -- ``model5f_lstm.py`` is byte
    identical -- trained against a velocity target, so the decoder emits ft/s
    per step where the position model emits ft.  Integrating it here is the
    whole difference; everything downstream (``smooth_constant_acceleration``,
    the NGSIM-to-world transform, the finite-difference speed) is inherited and
    sees displacement in ft either way.  ``predictor/stdan_vel`` does the same
    ``cumsum(v * dt)`` for the STDAN velocity checkpoint.
    """

    def __init__(self, ckpt_path: Optional[str] = None, **kwargs):
        super().__init__(ckpt_path=ckpt_path or str(DEFAULT_CKPT), **kwargs)

    def _run_model(self, tensors) -> np.ndarray:
        velocity_ft_s = super()._run_model(tensors)
        return np.cumsum(np.asarray(velocity_ft_s, dtype=np.float64) * self.dt, axis=0)
