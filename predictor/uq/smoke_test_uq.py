"""Smoke test for the conformal-novelty detector and the deep-ensemble members.

Run from the repository root:
    python predictor/uq/smoke_test_uq.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from predictor.stdan_vel.acc_adapter import STDANVelACCAdapter  # noqa: E402
from predictor.uq.conformal_ood import load_novelty  # noqa: E402
from predictor.uq.stream_tap import StdanNovelty  # noqa: E402

MEMBERS = [os.path.join(HERE, "ckpt", "ensemble", "member%d" % i, "best_model.pt")
           for i in range(1, 6)]


def main():
    detector = load_novelty()
    assert [detector.mondrian.n_of(g) for g in (0, 1, 2)] == [71568, 3357, 1677]

    # the stdan_vel default checkpoint is ensemble member0
    primary = STDANVelACCAdapter(mc_dropout=False)
    novelty = StdanNovelty(primary.gdEncoder, primary.generator, detector)

    steps = primary.full_length
    t = np.arange(steps, dtype=float) * primary.dt
    trackings = {
        1: np.column_stack((10.0 * t, np.zeros(steps), np.zeros(steps))),
        2: np.column_stack((10.0 * t + 8.0, np.ones(steps) * 3.5, np.zeros(steps))),
        3: np.column_stack((10.0 * t - 8.0, np.ones(steps) * -3.5, np.zeros(steps))),
    }
    raw = primary.predict_raw([1], trackings)[1]
    result = novelty.result()
    assert result["p"].shape == (1,)
    assert 0.0 < result["p"][0] <= 1.0
    assert result["group"][0] in (0, 1, 2)

    ends = []
    for path in [None] + MEMBERS:
        member = STDANVelACCAdapter(ckpt_path=path, mc_dropout=False)
        pred = member.predict_raw([1], trackings)[1]
        assert pred["raw_traj_xy"].shape == (3, 50, 2)
        assert np.isfinite(pred["raw_traj_xy"]).all()
        assert np.isclose(pred["raw_intention_prob"].sum(), 1.0)
        k = int(np.argmax(pred["raw_intention_prob"]))
        ends.append(pred["raw_traj_xy"][k, -1])
    ends = np.asarray(ends)
    spread = np.mean([np.linalg.norm(ends[i] - ends[j])
                      for i in range(len(ends)) for j in range(i + 1, len(ends))])
    assert spread > 1e-3

    print("novelty: p=%.4f p_cc=%.4f score=%.1f group=%d"
          % (result["p"][0], result["p_cc"][0], result["score"][0], result["group"][0]))
    print("primary intention prob (LK, LLC, RLC):",
          np.round(raw["raw_intention_prob"], 3).tolist())
    print("member0..5 end-point spread of the most likely mode at 5 s: %.3f m" % spread)


if __name__ == "__main__":
    main()
