from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from utils.acc_nair_smpc import MultimodalLeadPrediction, PolicySharingTree
except ImportError:  # pragma: no cover - used when imported from repository root
    from scripts.carla.utils.acc_nair_smpc import MultimodalLeadPrediction, PolicySharingTree


INTENTION_NAMES = ("LK", "LLC", "RLC")
REL_EGO_LANE = "ego_lane"
REL_LEFT_ADJACENT = "left_adjacent"
REL_RIGHT_ADJACENT = "right_adjacent"
REL_OTHER = "other"


@dataclass
class ACCModePrediction:
    vehicle_id: int
    mode_name: str
    probability: float
    frenet: np.ndarray
    active_mask: np.ndarray
    raw_mode_indices: Sequence[int]
    ego_lane_membership_split_step: int
    lead_covariance: Optional[np.ndarray] = None

    @property
    def branch_step(self) -> int:
        """Compatibility alias for older diagnostics."""
        return self.ego_lane_membership_split_step


@dataclass
class ACCProcessedPrediction:
    vehicle_id: int
    relation_to_ego_lane: str
    mode_predictions: List[ACCModePrediction]
    raw_mode_prob: Dict[str, float]
    acc_mode_prob: Dict[str, float]
    branch_info: Dict[str, object]
    raw_prediction: Optional[dict] = None


def normalize_probabilities(values: Sequence[float]) -> np.ndarray:
    probs = np.asarray(values, dtype=float).reshape(-1)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.ones_like(probs) / max(probs.size, 1)
    return probs / total


def reconstruct_acc_probabilities(raw_probs: Sequence[float], relation_to_ego_lane: str) -> Dict[str, object]:
    raw = normalize_probabilities(raw_probs)
    raw_map = {name: float(raw[idx]) for idx, name in enumerate(INTENTION_NAMES)}
    if relation_to_ego_lane == REL_LEFT_ADJACENT:
        acc = {"lk": raw_map["LK"] + raw_map["LLC"], "cutin": raw_map["RLC"], "cutout": 0.0}
    elif relation_to_ego_lane == REL_RIGHT_ADJACENT:
        acc = {"lk": raw_map["LK"] + raw_map["RLC"], "cutin": raw_map["LLC"], "cutout": 0.0}
    elif relation_to_ego_lane == REL_EGO_LANE:
        acc = {"lk": raw_map["LK"], "cutin": 0.0, "cutout": raw_map["LLC"] + raw_map["RLC"]}
    else:
        acc = {"lk": 1.0, "cutin": 0.0, "cutout": 0.0}
    norm = normalize_probabilities([acc["lk"], acc["cutin"], acc["cutout"]])
    acc = {"lk": float(norm[0]), "cutin": float(norm[1]), "cutout": float(norm[2])}
    return {"raw_mode_prob": raw_map, "acc_mode_prob": acc}


def relation_from_waypoints(ego_waypoint, target_waypoint) -> str:
    if ego_waypoint is None or target_waypoint is None:
        return REL_OTHER
    if target_waypoint.lane_id == ego_waypoint.lane_id and target_waypoint.road_id == ego_waypoint.road_id:
        return REL_EGO_LANE

    left = ego_waypoint.get_left_lane()
    if left is not None and left.lane_id == target_waypoint.lane_id and left.road_id == target_waypoint.road_id:
        return REL_LEFT_ADJACENT

    right = ego_waypoint.get_right_lane()
    if right is not None and right.lane_id == target_waypoint.lane_id and right.road_id == target_waypoint.road_id:
        return REL_RIGHT_ADJACENT
    return REL_OTHER


def global_to_frenet_trajectory(points_xy: np.ndarray, frenet_handler, yaw: float = 0.0) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    frenet = np.zeros((points.shape[0], 2), dtype=float)
    for idx, point in enumerate(points):
        s_val, d_val, _ = frenet_handler.convert_global_to_frenet_frame(
            float(point[0]), float(point[1]), float(yaw)
        )
        frenet[idx] = [s_val, d_val]
    return frenet


def ego_pose_relative_frenet(points_xy: np.ndarray, ego_x: float, ego_y: float, ego_yaw: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    dx = points[:, 0] - float(ego_x)
    dy = points[:, 1] - float(ego_y)
    c = np.cos(float(ego_yaw))
    s = np.sin(float(ego_yaw))
    frenet_s = c * dx + s * dy
    frenet_d = -s * dx + c * dy
    return np.stack((frenet_s, frenet_d), axis=-1)


def lane_occupancy_from_d(frenet: np.ndarray, ego_lane_d: float = 0.0, threshold: float = 0.5) -> np.ndarray:
    return np.abs(np.asarray(frenet, dtype=float)[:, 1] - float(ego_lane_d)) <= float(threshold)


def ego_lane_membership_split_step_from_mask(mode_memberships: np.ndarray) -> int:
    memberships = np.asarray(mode_memberships)
    if memberships.ndim != 2:
        raise ValueError("mode_memberships must have shape (num_modes, horizon+1)")
    reference = memberships[0]
    for step in range(memberships.shape[1]):
        if np.any(memberships[:, step] != reference[step]):
            return int(step)
    return int(memberships.shape[1] - 1)


def branch_step_from_lane_membership(mode_memberships: np.ndarray) -> int:
    """Compatibility alias for older tests/scripts."""
    return ego_lane_membership_split_step_from_mask(mode_memberships)


def _normalized_partition(labels: Sequence[object]) -> tuple[int, ...]:
    lookup = {}
    partition = []
    for label in labels:
        if label not in lookup:
            lookup[label] = len(lookup)
        partition.append(lookup[label])
    return tuple(partition)


def build_policy_sharing_tree_from_effective_leads(
        effective_lead_keys_by_scenario: Sequence[Sequence[tuple]],
        horizon: int,
        mode_names: Optional[List[str]] = None) -> PolicySharingTree:
    horizon = int(horizon)
    num_scenarios = len(effective_lead_keys_by_scenario)
    policy_groups = np.zeros((num_scenarios, horizon), dtype=int)
    if num_scenarios == 0:
        return PolicySharingTree(
            policy_groups_by_step=policy_groups,
            policy_split_steps=(),
            mode_names=mode_names,
        )

    next_group_id = 0
    previous_partition = None
    split_steps = []
    for step in range(horizon):
        prefix_to_group = {}
        prefixes = []
        for scenario_idx, keys in enumerate(effective_lead_keys_by_scenario):
            if len(keys) <= step:
                raise ValueError("effective lead key sequence is shorter than horizon")
            prefix = tuple(tuple(key) for key in keys[:step + 1])
            prefixes.append(prefix)
            if prefix not in prefix_to_group:
                prefix_to_group[prefix] = next_group_id
                next_group_id += 1
            policy_groups[scenario_idx, step] = prefix_to_group[prefix]

        partition = _normalized_partition(prefixes)
        if previous_partition is None:
            if len(set(partition)) > 1:
                split_steps.append(step)
        elif partition != previous_partition:
            split_steps.append(step)
        previous_partition = partition

    return PolicySharingTree(
        policy_groups_by_step=policy_groups,
        policy_split_steps=tuple(split_steps),
        mode_names=mode_names,
    )


def ego_lane_interval(mask: Sequence[bool]) -> tuple[int, int]:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if indices.size == 0:
        return 999, -1
    return int(indices[0]), int(indices[-1])


def _representative_index(raw_probs: np.ndarray, indices: Sequence[int]) -> int:
    return int(max(indices, key=lambda idx: raw_probs[idx]))


def _mode_specs_for_relation(relation: str, raw_probs: np.ndarray) -> List[tuple[str, List[int], float]]:
    if relation == REL_LEFT_ADJACENT:
        return [
            ("lk", [0, 1], float(raw_probs[0] + raw_probs[1])),
            ("cutin", [2], float(raw_probs[2])),
        ]
    if relation == REL_RIGHT_ADJACENT:
        return [
            ("lk", [0, 2], float(raw_probs[0] + raw_probs[2])),
            ("cutin", [1], float(raw_probs[1])),
        ]
    if relation == REL_EGO_LANE:
        return [
            ("lk", [0], float(raw_probs[0])),
            ("cutout", [1, 2], float(raw_probs[1] + raw_probs[2])),
        ]
    return [("lk", [0], 1.0)]


def _trajectory_aware_mode_specs(
    relation: str,
    raw_probs: np.ndarray,
    all_frenet: np.ndarray,
    ego_lane_d: float,
) -> tuple[List[tuple[str, List[int], float]], Optional[int]]:
    if relation not in (REL_LEFT_ADJACENT, REL_RIGHT_ADJACENT):
        return _mode_specs_for_relation(relation, raw_probs), None
    if all_frenet.ndim != 3 or all_frenet.shape[0] < 3:
        return _mode_specs_for_relation(relation, raw_probs), None

    candidate_indices = [1, 2]
    distances = []
    for idx in candidate_indices:
        d_values = all_frenet[idx, :, 1]
        initial_abs = abs(float(d_values[0]) - float(ego_lane_d))
        min_abs = float(np.min(np.abs(d_values - float(ego_lane_d))))
        final_abs = abs(float(d_values[-1]) - float(ego_lane_d))
        distances.append((min_abs, final_abs, -initial_abs, idx))
    _, _, _, cutin_idx = min(distances)
    lk_indices = [idx for idx in range(min(3, raw_probs.size)) if idx != cutin_idx]
    return [
        ("lk", lk_indices, float(np.sum(raw_probs[lk_indices]))),
        ("cutin", [cutin_idx], float(raw_probs[cutin_idx])),
    ], int(cutin_idx)


def _acc_probabilities_from_mode_specs(
    raw_probs: np.ndarray,
    raw_map: Dict[str, float],
    mode_specs: Sequence[tuple[str, List[int], float]],
) -> Dict[str, object]:
    acc = {"lk": 0.0, "cutin": 0.0, "cutout": 0.0}
    for mode_name, _, probability in mode_specs:
        if mode_name in acc:
            acc[mode_name] += float(probability)
    norm = normalize_probabilities([acc["lk"], acc["cutin"], acc["cutout"]])
    return {
        "raw_mode_prob": raw_map,
        "acc_mode_prob": {
            "lk": float(norm[0]),
            "cutin": float(norm[1]),
            "cutout": float(norm[2]),
        },
    }


def gate_unlikely_cutin_modes(
    mode_predictions: Sequence[ACCModePrediction],
    threshold: float,
) -> List[str]:
    """Stop unlikely cut-in modes from generating collision-avoidance constraints.

    Mirrors the vanishing chance constraint of Benciolini et al. (T-IV 2023):
    a candidate trajectory whose estimated probability is negligible should not
    restrict the ego vehicle.  Clearing ``active_mask`` is the 1D ACC equivalent
    of the safety ellipse collapsing to a point, since the controller skips every
    mode/step whose mask is unset.

    Only ``cutin`` modes are gated.  They exist solely for adjacent-lane targets,
    so an ego-lane lead vehicle can never be relaxed away.  A target already
    occupying the ego lane also keeps its step-0 constraint, because that step is
    taken from the measured position and is therefore shared by the ``lk`` mode.
    """
    threshold = float(threshold)
    if threshold <= 0.0:
        return []
    gated = []
    for mode in mode_predictions:
        if mode.mode_name == "cutin" and mode.probability < threshold:
            mode.active_mask[:] = False
            gated.append(mode.mode_name)
    return gated


def process_vehicle_prediction(
    raw_prediction: dict,
    relation_to_ego_lane: str,
    current_frenet_state: Sequence[float],
    horizon: int,
    dt: float,
    ego_lane_d: float = 0.0,
    ego_lane_threshold: float = 0.5,
    use_trajectory_aware_cutin_mapping: bool = False,
    mode_lane_memberships: Optional[np.ndarray] = None,
    lane_membership_source: str = "frenet_d_threshold",
    cutin_probability_threshold: float = 0.0,
) -> ACCProcessedPrediction:
    raw_probs = normalize_probabilities(raw_prediction["raw_intention_prob"])
    raw_map = {name: float(raw_probs[idx]) for idx, name in enumerate(INTENTION_NAMES)}
    traj_frenet = np.asarray(raw_prediction["pred_traj_frenet"], dtype=float)
    raw_vel = np.asarray(raw_prediction.get("raw_pred_vel", np.zeros_like(traj_frenet)), dtype=float)
    raw_lead_covariance = raw_prediction.get("pred_lead_covariance")
    lead_covariances = None
    vehicle_id = int(raw_prediction["vehicle_id"])

    horizon = int(horizon)
    steps = min(horizon + 1, traj_frenet.shape[1] + 1)
    current = np.asarray(current_frenet_state, dtype=float)
    current_s, current_d, current_v = float(current[0]), float(current[1]), float(current[2])

    all_frenet = np.zeros((traj_frenet.shape[0], steps, 3), dtype=float)
    memberships = np.zeros((traj_frenet.shape[0], steps), dtype=bool)
    if raw_lead_covariance is not None:
        sampled_covariance = np.asarray(raw_lead_covariance, dtype=float)
        expected_prefix = (traj_frenet.shape[0], steps - 1)
        if (
                sampled_covariance.ndim == 4
                and sampled_covariance.shape[:2] == expected_prefix
                and sampled_covariance.shape[2:] == (2, 2)):
            lead_covariances = np.zeros((traj_frenet.shape[0], steps, 2, 2), dtype=float)
            lead_covariances[:, 1:, :, :] = sampled_covariance[:, :steps - 1, :, :]
    provided_memberships = None
    if mode_lane_memberships is not None:
        provided_memberships = np.asarray(mode_lane_memberships, dtype=bool)
        expected = (traj_frenet.shape[0], steps)
        if provided_memberships.shape != expected:
            raise ValueError(
                f"mode_lane_memberships must have shape {expected}, "
                f"got {provided_memberships.shape}"
            )
    for mode in range(traj_frenet.shape[0]):
        mode_sd = np.vstack((np.array([[current_s, current_d]]), traj_frenet[mode, : steps - 1, :2]))
        speeds = np.zeros(steps, dtype=float)
        speeds[0] = current_v
        if raw_vel.ndim == 3 and raw_vel.shape[0] > mode:
            speeds[1:] = np.linalg.norm(raw_vel[mode, : steps - 1, :2], axis=1)
        elif steps > 1:
            speeds[1:] = np.linalg.norm(np.diff(mode_sd, axis=0), axis=1) / float(dt)
        all_frenet[mode, :, 0:2] = mode_sd
        all_frenet[mode, :, 2] = speeds
        if provided_memberships is None:
            memberships[mode] = lane_occupancy_from_d(mode_sd, ego_lane_d, ego_lane_threshold)
        else:
            memberships[mode] = provided_memberships[mode]

    ego_lane_membership_split_step = ego_lane_membership_split_step_from_mask(memberships)
    trajectory_mode_specs, trajectory_cutin_raw_idx = _trajectory_aware_mode_specs(
        relation_to_ego_lane,
        raw_probs,
        all_frenet,
        ego_lane_d,
    )
    mode_specs = (
        trajectory_mode_specs
        if use_trajectory_aware_cutin_mapping
        else _mode_specs_for_relation(relation_to_ego_lane, raw_probs)
    )
    prob_maps = _acc_probabilities_from_mode_specs(raw_probs, raw_map, mode_specs)
    mode_predictions = []
    for mode_name, raw_indices, probability in mode_specs:
        rep_idx = _representative_index(raw_probs, raw_indices)
        active = memberships[rep_idx].copy()
        if probability <= 0.0:
            continue
        mode_predictions.append(
            ACCModePrediction(
                vehicle_id=vehicle_id,
                mode_name=mode_name,
                probability=float(probability),
                frenet=all_frenet[rep_idx].copy(),
                active_mask=active,
                raw_mode_indices=list(raw_indices),
                ego_lane_membership_split_step=ego_lane_membership_split_step,
                lead_covariance=(
                    None if lead_covariances is None else lead_covariances[rep_idx].copy()
                ),
            )
        )

    if mode_predictions:
        norm = normalize_probabilities([mode.probability for mode in mode_predictions])
        for idx, mode in enumerate(mode_predictions):
            mode.probability = float(norm[idx])

    gated_cutin_modes = gate_unlikely_cutin_modes(
        mode_predictions, cutin_probability_threshold
    )

    start_idx, end_idx = ego_lane_interval(np.any(memberships, axis=0))
    return ACCProcessedPrediction(
        vehicle_id=vehicle_id,
        relation_to_ego_lane=relation_to_ego_lane,
        mode_predictions=mode_predictions,
        raw_mode_prob=prob_maps["raw_mode_prob"],
        acc_mode_prob=prob_maps["acc_mode_prob"],
        branch_info={
            "ego_lane_membership_split_step": ego_lane_membership_split_step,
            "branch_step": ego_lane_membership_split_step,
            "mode_lane_membership": memberships,
            "ego_lane_occupancy_mask": np.any(memberships, axis=0),
            "ego_lane_start_idx": start_idx,
            "ego_lane_end_idx": end_idx,
            "lane_membership_source": lane_membership_source,
            "trajectory_suggested_cutin_raw_idx": trajectory_cutin_raw_idx,
            "used_trajectory_aware_cutin_mapping": bool(use_trajectory_aware_cutin_mapping),
            "cutin_probability_threshold": float(cutin_probability_threshold),
            "gated_cutin_modes": gated_cutin_modes,
        },
        raw_prediction=raw_prediction,
    )


def build_multitarget_lead_prediction(
    processed_predictions: Iterable[ACCProcessedPrediction],
    ego_state: Sequence[float],
    horizon: int,
    desired_speed: float,
    num_modes: int,
    covariance: Optional[np.ndarray] = None,
    nonblocking_speed: Optional[float] = None,
) -> tuple[MultimodalLeadPrediction, List[dict]]:
    targets = [pred for pred in processed_predictions if pred.mode_predictions]
    horizon = int(horizon)
    num_modes = int(num_modes)
    ego_s = float(np.asarray(ego_state, dtype=float)[0])
    default_speed = float(desired_speed if nonblocking_speed is None else nonblocking_speed)
    if not targets:
        means = np.zeros((1, horizon + 1, 2), dtype=float)
        means[:, :, 0] = ego_s
        means[:, :, 1] = default_speed
        active_mask = np.zeros((1, horizon + 1), dtype=bool)
        prediction = MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0]),
            covariances=np.zeros((1, horizon + 1, 2, 2)),
            mode_names=["default"],
            active_mask=active_mask,
        )
        return prediction, []

    combos = list(itertools.product(*[target.mode_predictions for target in targets]))
    scenarios = []
    for combo_idx, combo in enumerate(combos):
        prob = float(np.prod([mode.probability for mode in combo]))
        scenarios.append((combo_idx, combo, prob))
    scenarios.sort(key=lambda item: item[2], reverse=True)
    scenarios = scenarios[:num_modes]

    means = np.zeros((len(scenarios), horizon + 1, 2), dtype=float)
    covariances = np.zeros((len(scenarios), horizon + 1, 2, 2), dtype=float)
    active_mask = np.zeros((len(scenarios), horizon + 1), dtype=bool)
    mode_names = []
    probabilities = []
    metadata = []
    effective_lead_keys_by_scenario = []
    base_cov = np.asarray(covariance if covariance is not None else np.diag([0.25, 0.09]), dtype=float)

    for scenario_out_idx, (combo_idx, combo, prob) in enumerate(scenarios):
        means[scenario_out_idx, :, 0] = ego_s
        means[scenario_out_idx, :, 1] = default_speed
        mode_names.append("+".join(f"{mode.vehicle_id}:{mode.mode_name}" for mode in combo))
        probabilities.append(prob)
        selected_ids = []
        selected_modes = []
        effective_lead_keys = []
        for step in range(horizon + 1):
            candidates = []
            for mode in combo:
                if step >= mode.frenet.shape[0] or not mode.active_mask[step]:
                    continue
                s_val = float(mode.frenet[step, 0])
                if s_val < ego_s:
                    continue
                candidates.append((s_val, mode))
            if not candidates:
                selected_ids.append(None)
                selected_modes.append(None)
                effective_lead_keys.append(("inactive",))
                continue
            _, selected = min(candidates, key=lambda item: item[0])
            means[scenario_out_idx, step, 0] = selected.frenet[step, 0]
            means[scenario_out_idx, step, 1] = selected.frenet[step, 2]
            if selected.lead_covariance is not None and step < selected.lead_covariance.shape[0]:
                covariances[scenario_out_idx, step] = selected.lead_covariance[step]
            else:
                covariances[scenario_out_idx, step] = base_cov
            active_mask[scenario_out_idx, step] = True
            selected_ids.append(selected.vehicle_id)
            selected_modes.append(selected.mode_name)
            effective_lead_keys.append((int(selected.vehicle_id), str(selected.mode_name)))
        effective_lead_keys_by_scenario.append(effective_lead_keys)
        metadata.append(
            {
                "scenario_id": combo_idx,
                "probability": prob,
                "mode_names": mode_names[-1],
                "selected_vehicle_ids": selected_ids,
                "selected_mode_names": selected_modes,
                "effective_lead_keys": effective_lead_keys,
            }
        )

    policy_tree = build_policy_sharing_tree_from_effective_leads(
        effective_lead_keys_by_scenario,
        horizon=horizon,
        mode_names=mode_names,
    )
    k_group_map = np.zeros((len(scenarios), horizon), dtype=int)
    k_group_names = []
    k_group_lookup = {}
    for scenario_idx, effective_lead_keys in enumerate(effective_lead_keys_by_scenario):
        for step in range(horizon):
            key = tuple(effective_lead_keys[step])
            if key not in k_group_lookup:
                k_group_lookup[key] = len(k_group_names)
                k_group_names.append(":".join(str(part) for part in key))
            k_group_map[scenario_idx, step] = k_group_lookup[key]
    prediction = MultimodalLeadPrediction(
        means=means,
        probabilities=np.asarray(probabilities, dtype=float),
        covariances=covariances,
        mode_names=mode_names,
        policy_tree=policy_tree,
        active_mask=active_mask,
        k_group_map=k_group_map,
        k_group_names=k_group_names,
    )
    return prediction, metadata
