from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


TrackId = Union[int, str]


SENSOR_CONFIG = {
    "front_long_radar": {
        "range_m": 160.0,
        "angle_center_deg": 0.0,
        "half_angle_deg": 12.0,
        "enabled": True,
    },
    "front_camera": {
        "range_m": 120.0,
        "angle_center_deg": 0.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "front_near_radar": {
        "range_m": 60.0,
        "angle_center_deg": 0.0,
        "half_angle_deg": 45.0,
        "enabled": True,
    },
    "front_left_corner_radar": {
        "range_m": 45.0,
        "angle_center_deg": 45.0,
        "half_angle_deg": 45.0,
        "enabled": True,
    },
    "front_right_corner_radar": {
        "range_m": 45.0,
        "angle_center_deg": -45.0,
        "half_angle_deg": 45.0,
        "enabled": True,
    },
    "rear_left_corner_radar": {
        "range_m": 45.0,
        "angle_center_deg": 135.0,
        "half_angle_deg": 45.0,
        "enabled": True,
    },
    "rear_right_corner_radar": {
        "range_m": 45.0,
        "angle_center_deg": -135.0,
        "half_angle_deg": 45.0,
        "enabled": True,
    },
    "ultrasonic_00": {
        "range_m": 5.0,
        "angle_center_deg": -165.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_01": {
        "range_m": 5.0,
        "angle_center_deg": -135.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_02": {
        "range_m": 5.0,
        "angle_center_deg": -105.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_03": {
        "range_m": 5.0,
        "angle_center_deg": -75.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_04": {
        "range_m": 5.0,
        "angle_center_deg": -45.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_05": {
        "range_m": 5.0,
        "angle_center_deg": -15.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_06": {
        "range_m": 5.0,
        "angle_center_deg": 15.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_07": {
        "range_m": 5.0,
        "angle_center_deg": 45.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_08": {
        "range_m": 5.0,
        "angle_center_deg": 75.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_09": {
        "range_m": 5.0,
        "angle_center_deg": 105.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_10": {
        "range_m": 5.0,
        "angle_center_deg": 135.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
    "ultrasonic_11": {
        "range_m": 5.0,
        "angle_center_deg": 165.0,
        "half_angle_deg": 20.0,
        "enabled": True,
    },
}


NOISE_CONFIG = {
    "clean": {
        "sigma_s_m": 0.2,
        "sigma_d_m": 0.1,
        "sigma_v_mps": 0.1,
    },
    "nominal": {
        "sigma_s_m": 0.5,
        "sigma_d_m": 0.2,
        "sigma_v_mps": 0.3,
    },
    "adverse": {
        "sigma_s_m": 1.0,
        "sigma_d_m": 0.5,
        "sigma_v_mps": 0.8,
    },
}


@dataclass(frozen=True)
class Track:
    id: TrackId
    x: float
    y: float
    yaw: float
    v: float
    x_rel: float
    y_rel: float
    distance: float
    bearing_deg: float
    s: Optional[float]
    d: Optional[float]
    lane_id: Optional[TrackId]
    sigma_s: float
    sigma_d: float
    sigma_v: float
    visible_sensor_names: List[str]
    source: str
    timestamp: float


@dataclass(frozen=True)
class _ActorState:
    id: TrackId
    x: float
    y: float
    yaw: float
    v: float
    s: Optional[float] = None
    d: Optional[float] = None
    lane_id: Optional[TrackId] = None


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def transform_to_ego_frame(ego_state: Any, actor_state: Any) -> Tuple[float, float, float, float]:
    ego = _to_actor_state(ego_state)
    actor = _to_actor_state(actor_state)

    dx = actor.x - ego.x
    dy = actor.y - ego.y

    x_rel = cos(ego.yaw) * dx + sin(ego.yaw) * dy
    y_rel = -sin(ego.yaw) * dx + cos(ego.yaw) * dy
    distance = hypot(x_rel, y_rel)
    bearing_deg = degrees(atan2(y_rel, x_rel))

    return x_rel, y_rel, distance, bearing_deg


def get_visible_sensor_names(
    ego_state: Any,
    actor_state: Any,
    sensor_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    _, _, distance, bearing_deg = transform_to_ego_frame(ego_state, actor_state)
    sensors = SENSOR_CONFIG if sensor_config is None else sensor_config

    visible_sensor_names = []
    for sensor_name, sensor in sensors.items():
        if not sensor.get("enabled", True):
            continue
        if distance > sensor["range_m"]:
            continue
        angle_error = abs(angle_diff_deg(bearing_deg, sensor["angle_center_deg"]))
        if angle_error <= sensor["half_angle_deg"]:
            visible_sensor_names.append(sensor_name)

    return visible_sensor_names


class TrackLevelUncertaintyWrapper:
    def __init__(
        self,
        sensor_config: Optional[Dict[str, Dict[str, Any]]] = None,
        noise_config: Optional[Dict[str, Dict[str, float]]] = None,
        noise_level: str = "nominal",
        seed: int = 0,
    ):
        self.sensor_config = deepcopy(SENSOR_CONFIG if sensor_config is None else sensor_config)
        self.noise_config = deepcopy(NOISE_CONFIG if noise_config is None else noise_config)
        if noise_level not in self.noise_config:
            raise ValueError(f"Unknown noise level: {noise_level}")
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def update(
        self,
        ego_state: Any,
        gt_actors: List[Any],
        lane_map: Any = None,
        timestamp: float = 0.0,
    ) -> Tuple[List[Track], Dict[str, Any]]:
        del lane_map
        ego = _to_actor_state(ego_state)
        observed_tracks = []
        dropped_actor_ids = []
        drop_reasons = {}

        for gt_actor in gt_actors:
            actor = _to_actor_state(gt_actor)
            visible_sensor_names = get_visible_sensor_names(ego, actor, self.sensor_config)

            if not visible_sensor_names:
                dropped_actor_ids.append(actor.id)
                drop_reasons[actor.id] = "out_of_range_or_fov"
                continue

            observed_tracks.append(
                self._make_noisy_track(ego, actor, visible_sensor_names, timestamp)
            )

        log_dict = {
            "num_gt_actors": len(gt_actors),
            "num_observed_tracks": len(observed_tracks),
            "num_dropped_tracks": len(dropped_actor_ids),
            "dropped_actor_ids": dropped_actor_ids,
            "drop_reasons": drop_reasons,
            "noise_level": self.noise_level,
        }

        return observed_tracks, log_dict

    def _make_noisy_track(
        self,
        ego: _ActorState,
        actor: _ActorState,
        visible_sensor_names: List[str],
        timestamp: float,
    ) -> Track:
        noise = self.noise_config[self.noise_level]
        sigma_s = noise["sigma_s_m"]
        sigma_d = noise["sigma_d_m"]
        sigma_v = noise["sigma_v_mps"]

        x_rel, y_rel, _, _ = transform_to_ego_frame(ego, actor)
        s_noisy = None if actor.s is None else actor.s + self.rng.normal(0.0, sigma_s)
        d_noisy = None if actor.d is None else actor.d + self.rng.normal(0.0, sigma_d)

        if actor.s is None or actor.d is None:
            x_rel = x_rel + self.rng.normal(0.0, sigma_s)
            y_rel = y_rel + self.rng.normal(0.0, sigma_d)

        distance = hypot(x_rel, y_rel)
        bearing_deg = degrees(atan2(y_rel, x_rel))
        x_obs, y_obs = _ego_to_world_xy(ego, x_rel, y_rel)
        v_noisy = actor.v + self.rng.normal(0.0, sigma_v)

        return Track(
            id=actor.id,
            x=x_obs,
            y=y_obs,
            yaw=actor.yaw,
            v=v_noisy,
            x_rel=x_rel,
            y_rel=y_rel,
            distance=distance,
            bearing_deg=bearing_deg,
            s=s_noisy,
            d=d_noisy,
            lane_id=actor.lane_id,
            sigma_s=sigma_s,
            sigma_d=sigma_d,
            sigma_v=sigma_v,
            visible_sensor_names=visible_sensor_names,
            source="observed",
            timestamp=timestamp,
        )


def _ego_to_world_xy(ego: _ActorState, x_rel: float, y_rel: float) -> Tuple[float, float]:
    dx = cos(ego.yaw) * x_rel - sin(ego.yaw) * y_rel
    dy = sin(ego.yaw) * x_rel + cos(ego.yaw) * y_rel
    return ego.x + dx, ego.y + dy


def _to_actor_state(state: Any) -> _ActorState:
    if isinstance(state, _ActorState):
        return state

    if hasattr(state, "get_location") and hasattr(state, "get_transform"):
        return _carla_actor_to_state(state)

    return _plain_state_to_actor_state(state)


def _plain_state_to_actor_state(state: Any) -> _ActorState:
    actor_id = _read_field(state, "id", "unknown")
    x = float(_read_field(state, "x"))
    y = float(_read_field(state, "y"))
    yaw = _read_field(state, "yaw", 0.0)
    yaw = float(yaw)
    if abs(yaw) > 2.0 * pi:
        yaw = radians(yaw)

    v = _read_field(state, "v", None)
    if v is None:
        vx = float(_read_field(state, "vx", 0.0))
        vy = float(_read_field(state, "vy", 0.0))
        v = vx * cos(yaw) + vy * sin(yaw)

    return _ActorState(
        id=actor_id,
        x=x,
        y=y,
        yaw=yaw,
        v=float(v),
        s=_optional_float(_read_field(state, "s", None)),
        d=_optional_float(_read_field(state, "d", None)),
        lane_id=_read_field(state, "lane_id", None),
    )


def _carla_actor_to_state(actor: Any) -> _ActorState:
    loc = actor.get_location()
    tf = actor.get_transform()
    yaw = -radians(float(tf.rotation.yaw))

    v = 0.0
    if hasattr(actor, "get_velocity"):
        vel = actor.get_velocity()
        v = float(vel.x) * cos(yaw) - float(vel.y) * sin(yaw)

    return _ActorState(
        id=getattr(actor, "id", "unknown"),
        x=float(loc.x),
        y=-float(loc.y),
        yaw=yaw,
        v=v,
    )


def _read_field(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
