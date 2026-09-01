import os

import numpy as np

from predictor.stdan.stdan_predictor import STDANPredictor, carla_to_ngsim_history


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ckpt = os.path.join(root, "predictor", "stdan", "ckpt", "best_model.pt")
    predictor = STDANPredictor(ckpt, mc_dropout=False)

    converted = carla_to_ngsim_history(
        np.array([[0.0, 0.0], [2.0, 3.0]]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert np.allclose(converted, np.array([[0.0, 0.0], [3.0, 2.0]]))

    steps = predictor.full_length
    t = np.arange(steps, dtype=float) * predictor.dt
    trackings = {
        1: np.column_stack((10.0 * t, np.zeros(steps), np.zeros(steps))),
        2: np.column_stack((10.0 * t + 8.0, np.ones(steps) * 3.0, np.zeros(steps))),
        3: np.column_stack((10.0 * t - 8.0, np.ones(steps) * -3.0, np.zeros(steps))),
    }
    target_state = trackings[1][-1]

    tensors = predictor.make_tensor(1, target_state, trackings)
    hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls = tensors
    assert hist.shape == (31, 1, 2)
    assert nbrs.shape == (31, 2, 2)
    assert mask.shape == (1, 3, 13, 64)
    assert str(mask.dtype) == "torch.bool"
    assert va.shape == (31, 1, 2)
    assert nbrsva.shape == (31, 2, 2)
    assert lane.shape == (31, 1, 1)
    assert nbrslane.shape == (31, 2, 1)
    assert cls.shape == (31, 1, 1)
    assert nbrscls.shape == (31, 2, 1)

    result = predictor.predict(1, target_state, trackings, num_modes=2, horizon_steps=8)
    assert result["stdan_output"].shape == (9, 50, 5)
    assert result["probabilities"].shape == (9,)
    assert np.isclose(result["probabilities"].sum(), 1.0)
    assert result["mus"].shape == (2, 8, 2)
    assert result["sigmas"].shape == (2, 8, 2, 2)

    print("conversion:", converted.tolist())
    print("hist:", tuple(hist.shape), "nbrs:", tuple(nbrs.shape), "mask:", tuple(mask.shape))
    print("stdan_output:", result["stdan_output"].shape)
    print("mus:", result["mus"].shape, "sigmas:", result["sigmas"].shape)
    print("top_mode_indices:", result["top_mode_indices"].tolist())


if __name__ == "__main__":
    main()
