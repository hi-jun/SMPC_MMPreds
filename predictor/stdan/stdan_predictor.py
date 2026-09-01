"""
CARLA-facing STDAN predictor wrapper.

The model is trained with NGSIM target-centric axes:
  x: lateral, y: longitudinal, units: feet inside the network.

Public prediction output is normalized to:
  (9, 50, 5) = (mode, future step, [mean_x, std_x, mean_y, std_y, correlation])
with means/stds in meters in the target-centric NGSIM frame.  The wrapper also
converts that distribution into global CARLA/RHS GMM moments for the existing
SMPC code path.
"""

import os

import numpy as np
import torch

from .config import get_stdan_args
from .model5f_mult import GDEncoder, Generator


M2FT = 3.281
FT2M = 0.3048
STDAN_OUTPUT_SIZE = 5


def _rotation_matrix(yaw):
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float64,
    )


def transform_points(points, origin_x, origin_y, yaw, inverse=False):
    """Same rigid transform used by utils_STDAN.transformation."""
    points = np.asarray(points, dtype=np.float64)
    rotation = _rotation_matrix(yaw)
    if inverse:
        return points @ rotation.T + np.array([origin_x, origin_y], dtype=np.float64)
    return (points - np.array([origin_x, origin_y], dtype=np.float64)) @ rotation


def carla_to_ngsim_history(points_xy, target_state):
    """
    Convert CARLA/RHS global history to target-centered NGSIM model axes.

    This is intentionally applied immediately before tensor construction.  The
    incoming scenario history has already been converted from CARLA's left-handed
    actor frame to the repo's right-handed global frame by AgentHistory.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] < 2:
        raise ValueError(f"Expected history shaped (T, >=2), got {points_xy.shape}")

    target_x, target_y, target_yaw = target_state[:3]
    local_rhs = transform_points(points_xy[:, :2], target_x, target_y, target_yaw)
    return local_rhs[:, [1, 0]]


class STDANPredictor:
    def __init__(
        self,
        ckpt_path,
        history=3.0,
        future=5.0,
        dt=0.1,
        input_coordinate_system="carla",
        apply_carla_rlc_bias=False,
        mc_dropout=False,
        load_model=True,
    ):
        self.args = get_stdan_args(dt=dt, history=history, future=future)
        self.device = self.args["device"]
        self.dt = float(dt)
        self.history = float(history)
        self.future = float(future)
        self.in_length = self.args["in_length"]
        self.out_length = self.args["out_length"]
        self.full_length = self.in_length + 2
        self.encoder_size = self.args["lstm_encoder_size"]
        self.n_head = self.args["n_head"]
        self.grid_size = self.args["grid_size"]
        self.use_maneuvers = self.args["use_maneuvers"]
        self.num_modes = self.args["lat_length"] * self.args["lon_length"]
        self.input_coordinate_system = input_coordinate_system.lower()
        self.apply_carla_rlc_bias = bool(apply_carla_rlc_bias)
        self.mc_dropout = bool(mc_dropout)

        if self.input_coordinate_system not in {"carla", "ngsim"}:
            raise ValueError(
                "input_coordinate_system must be 'carla' or 'ngsim', "
                f"got {input_coordinate_system!r}"
            )

        self.gdEncoder = None
        self.generator = None
        if load_model:
            self._load_model(ckpt_path)

    def _load_model(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"STDAN checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch 1.12, used by mmpreds-c0913 in this workspace, does not
            # expose weights_only yet.
            ckpt = torch.load(ckpt_path, map_location=self.device)
        self.gdEncoder = GDEncoder(self.args).to(self.device)
        self.generator = Generator(self.args).to(self.device)
        self.gdEncoder.load_state_dict(ckpt["gdEncoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.gdEncoder.eval()
        self.generator.eval()

    def _empty_neighbor_tensors(self, n_objs=0):
        return (
            torch.zeros(self.in_length, n_objs, 2, device=self.device),
            torch.zeros(self.in_length, n_objs, 2, device=self.device),
            torch.zeros(self.in_length, n_objs, 1, device=self.device),
            torch.full((self.in_length, n_objs, 1), 2.0, device=self.device),
        )

    def _base_tensors(self):
        hist = torch.zeros(self.in_length, 1, 2, device=self.device)
        va = torch.zeros(self.in_length, 1, 2, device=self.device)
        mask = torch.zeros(
            1,
            self.grid_size[1],
            self.grid_size[0],
            self.encoder_size,
            device=self.device,
        )
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
        lat_edges = np.linspace(lat_min, lat_max, num_lat_cells + 1)
        lon_edges = np.linspace(lon_min, lon_max, num_lon_cells + 1)
        return lat_edges, lon_edges

    def _history_to_ngsim(self, history, target_state):
        history = np.asarray(history, dtype=np.float64)[-self.full_length:, :2]
        if self.input_coordinate_system == "carla":
            return carla_to_ngsim_history(history, target_state)
        return history

    def _motion_tensors_from_ngsim(self, ngsim_history_m):
        """Return hist and [speed, signed accel] tensors, preserving utils_STDAN math."""
        history_ft = torch.from_numpy(ngsim_history_m * M2FT).float().to(self.device)
        velocity = (history_ft[1:] - history_ft[:-1]) / self.dt
        acceleration = (velocity[1:] - velocity[:-1]) / self.dt

        hist = history_ft[-self.in_length :]
        final_velocity = velocity[-self.in_length :]
        speed = torch.norm(final_velocity, p=2, dim=-1, keepdim=True)
        dot_product = (acceleration * final_velocity).sum(dim=-1, keepdim=True)
        signed_acceleration = dot_product / (speed + 1e-8)
        va = torch.cat((speed, signed_acceleration), dim=-1)
        return hist, va

    def _append_valid_neighbor(
        self,
        valid_neighbors,
        mask_tsr,
        flat_grid_index,
        grid_lat,
        grid_lon,
        nbr_hist_tsr,
        nbr_va_tsr,
    ):
        valid_neighbors.append((flat_grid_index, nbr_hist_tsr, nbr_va_tsr))
        mask_tsr[0, grid_lat, grid_lon, :] = 1.0

    def _finalize_neighbors(self, valid_neighbors):
        if not valid_neighbors:
            return self._empty_neighbor_tensors()

        valid_neighbors.sort(key=lambda item: item[0])
        nbrs_tsr = torch.stack([item[1] for item in valid_neighbors], dim=1)
        nbrsva_tsr = torch.stack([item[2] for item in valid_neighbors], dim=1)
        n_objs = len(valid_neighbors)
        nbrslane_tsr = torch.zeros(self.in_length, n_objs, 1, device=self.device)
        nbrscls_tsr = torch.full((self.in_length, n_objs, 1), 2.0, device=self.device)
        return nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr

    def _assert_tensor_contract(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls):
        """Shape guard for the model's masked_scatter path."""
        n_objs = nbrs.shape[1]
        assert hist.shape == (self.in_length, 1, 2)
        assert va.shape == (self.in_length, 1, 2)
        assert mask.shape == (1, self.grid_size[1], self.grid_size[0], self.encoder_size)
        assert mask.dtype == torch.bool
        assert nbrs.shape == (self.in_length, n_objs, 2)
        assert nbrsva.shape == (self.in_length, n_objs, 2)
        assert lane.shape == (self.in_length, 1, 1)
        assert nbrslane.shape == (self.in_length, n_objs, 1)
        assert cls.shape == (self.in_length, 1, 1)
        assert nbrscls.shape == (self.in_length, n_objs, 1)

    def make_tensor(self, target_id, target_state, trackings, no_nbsr=False):
        """
        Build the 9 input tensors expected by GDEncoder.

        The flattening order is unchanged from utils_STDAN: valid neighbors are
        sorted by row-major social-grid index (lat row, then lon column), matching
        the mask order used by GDEncoder.masked_scatter_.
        """
        hist_tsr, va_tsr, mask_tsr, lane_tsr, cls_tsr = self._base_tensors()

        if target_id not in trackings or len(trackings[target_id]) < self.full_length:
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = self._empty_neighbor_tensors()
            tensors = (
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
            self._assert_tensor_contract(*tensors)
            return tensors

        target_ngsim = self._history_to_ngsim(trackings[target_id], target_state)
        hist_tsr[:, 0, :], va_tsr[:, 0, :] = self._motion_tensors_from_ngsim(target_ngsim)

        if no_nbsr:
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = self._empty_neighbor_tensors()
            tensors = (
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
            self._assert_tensor_contract(*tensors)
            return tensors

        lat_edges, lon_edges = self._grid_edges()
        valid_neighbors = []

        for nbr_id in [agent_id for agent_id in trackings.keys() if agent_id != target_id]:
            if len(trackings[nbr_id]) < self.full_length:
                continue

            nbr_ngsim = self._history_to_ngsim(trackings[nbr_id], target_state)
            nbr_hist_tsr, nbr_va_tsr = self._motion_tensors_from_ngsim(nbr_ngsim)
            nbr_current = nbr_ngsim[-1]

            grid_lat = np.digitize(nbr_current[0], lat_edges) - 1
            grid_lon = np.digitize(nbr_current[1], lon_edges) - 1

            if 0 <= grid_lat < self.grid_size[1] and 0 <= grid_lon < self.grid_size[0]:
                flat_grid_index = grid_lat * self.grid_size[0] + grid_lon
                self._append_valid_neighbor(
                    valid_neighbors,
                    mask_tsr,
                    flat_grid_index,
                    grid_lat,
                    grid_lon,
                    nbr_hist_tsr,
                    nbr_va_tsr,
                )

        nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = self._finalize_neighbors(valid_neighbors)
        tensors = (
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
        self._assert_tensor_contract(*tensors)
        return tensors

    def _run_model(self, tensors):
        if self.gdEncoder is None or self.generator is None:
            raise RuntimeError("STDAN model is not loaded")

        hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls = tensors
        if self.use_maneuvers:
            lat_enc = torch.zeros(1, self.args["lat_length"], device=self.device)
            lon_enc = torch.zeros(1, self.args["lon_length"], device=self.device)
        else:
            lat_enc = torch.empty(1, 1, device=self.device)
            lon_enc = torch.empty(1, 1, device=self.device)

        self.gdEncoder.train() if self.mc_dropout else self.gdEncoder.eval()
        self.generator.eval()
        with torch.no_grad():
            values, spatial_weight, _, _ = self.gdEncoder(
                hist,
                nbrs,
                mask,
                va,
                nbrsva,
                lane,
                nbrslane,
                cls,
                nbrscls,
            )
            predictions, lat_predictions, lon_predictions = self.generator(values, lat_enc, lon_enc)
        self.gdEncoder.eval()
        return predictions, lat_predictions, lon_predictions, spatial_weight, mask

    def _stdan_output_from_model(self, predictions):
        model_output = np.asarray(predictions, dtype=np.float64)
        if model_output.ndim == 4 and model_output.shape[2] == 1:
            model_output = np.squeeze(model_output, axis=2)

        expected_shape = (self.num_modes, self.out_length, STDAN_OUTPUT_SIZE)
        assert model_output.shape == expected_shape, (
            f"STDAN model output shape {model_output.shape} != {expected_shape}"
        )

        stdan_output = np.empty_like(model_output)
        stdan_output[:, :, 0] = model_output[:, :, 0] * FT2M
        stdan_output[:, :, 1] = model_output[:, :, 2] * FT2M
        stdan_output[:, :, 2] = model_output[:, :, 1] * FT2M
        stdan_output[:, :, 3] = model_output[:, :, 3] * FT2M
        stdan_output[:, :, 4] = model_output[:, :, 4]

        if self.input_coordinate_system == "carla" and self.apply_carla_rlc_bias:
            rlc_indices = [2, 5, 8]
            stdan_output[rlc_indices, :, 0] += 0.5

        assert stdan_output.shape == expected_shape
        return stdan_output

    @staticmethod
    def _mode_probabilities(lat_predictions, lon_predictions):
        lat_probs = np.asarray(lat_predictions, dtype=np.float64).reshape(-1)
        lon_probs = np.asarray(lon_predictions, dtype=np.float64).reshape(-1)
        probabilities = np.outer(lon_probs, lat_probs).flatten()
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        return probabilities

    def stdan_output_to_global_gmm(self, stdan_output, target_state):
        """
        Convert target-centric NGSIM STDAN distributions to global CARLA/RHS GMM.

        Input order is public STDAN order:
          [mean_x, std_x, mean_y, std_y, correlation].
        """
        expected_shape = (self.num_modes, self.out_length, STDAN_OUTPUT_SIZE)
        assert stdan_output.shape == expected_shape, (
            f"STDAN output shape {stdan_output.shape} != {expected_shape}"
        )

        means_ngsim = np.stack((stdan_output[:, :, 0], stdan_output[:, :, 2]), axis=-1)
        means_local_rhs = means_ngsim[:, :, [1, 0]]
        means_global = transform_points(
            means_local_rhs.reshape(-1, 2),
            target_state[0],
            target_state[1],
            target_state[2],
            inverse=True,
        ).reshape(self.num_modes, self.out_length, 2)

        swap_ngsim_to_rhs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        rot_local_to_global = _rotation_matrix(target_state[2])
        sigmas_global = np.empty((self.num_modes, self.out_length, 2, 2), dtype=np.float64)

        std_x = stdan_output[:, :, 1]
        std_y = stdan_output[:, :, 3]
        rho = np.clip(stdan_output[:, :, 4], -0.999, 0.999)
        for mode_idx in range(self.num_modes):
            for step_idx in range(self.out_length):
                sx = max(float(std_x[mode_idx, step_idx]), 1e-4)
                sy = max(float(std_y[mode_idx, step_idx]), 1e-4)
                r = float(rho[mode_idx, step_idx])
                cov_ngsim = np.array(
                    [[sx * sx, r * sx * sy], [r * sx * sy, sy * sy]],
                    dtype=np.float64,
                )
                cov_rhs = swap_ngsim_to_rhs @ cov_ngsim @ swap_ngsim_to_rhs.T
                sigmas_global[mode_idx, step_idx] = (
                    rot_local_to_global @ cov_rhs @ rot_local_to_global.T
                )

        return means_global, sigmas_global

    @staticmethod
    def _select_top_modes(mus, sigmas, probabilities, num_modes):
        if num_modes is None:
            return mus, sigmas, probabilities, np.arange(len(probabilities))
        if num_modes > len(probabilities):
            raise ValueError(
                f"Requested {num_modes} modes, but STDAN only produces {len(probabilities)}"
            )

        top_indices = np.argsort(-probabilities)[:num_modes]
        top_probabilities = probabilities[top_indices]
        prob_sum = top_probabilities.sum()
        if prob_sum > 0:
            top_probabilities = top_probabilities / prob_sum
        return mus[top_indices], sigmas[top_indices], top_probabilities, top_indices

    def predict(self, target_id, target_state, trackings, num_modes=None, horizon_steps=None):
        """
        Return STDAN's full distribution plus the top-k global GMM used by SMPC.
        """
        tensors = self.make_tensor(target_id, target_state, trackings, no_nbsr=False)
        predictions, lat_pred, lon_pred, spatial_weight, mask = self._run_model(tensors)
        stdan_output = self._stdan_output_from_model(predictions)
        probabilities = self._mode_probabilities(lat_pred, lon_pred)

        expected_shape = (self.num_modes, self.out_length, STDAN_OUTPUT_SIZE)
        assert stdan_output.shape == expected_shape
        assert probabilities.shape == (self.num_modes,)

        mus, sigmas = self.stdan_output_to_global_gmm(stdan_output, target_state)
        if horizon_steps is not None:
            mus = mus[:, :horizon_steps, :]
            sigmas = sigmas[:, :horizon_steps, :, :]

        mus_top, sigmas_top, probs_top, top_indices = self._select_top_modes(
            mus, sigmas, probabilities, num_modes
        )

        return {
            "stdan_output": stdan_output,
            "probabilities": probabilities,
            "mus": mus_top,
            "sigmas": sigmas_top,
            "mode_probabilities": probs_top,
            "top_mode_indices": top_indices,
            "spatial_weight": spatial_weight,
            "mask": mask,
        }
