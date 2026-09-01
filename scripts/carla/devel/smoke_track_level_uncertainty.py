import os
import sys
from math import cos, radians, sin

import numpy as np


CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(CARLA_SCRIPT_DIR)

from perception.track_level_uncertainty import SENSOR_CONFIG, TrackLevelUncertaintyWrapper


def actor_at(actor_id, distance, bearing_deg, v=10.0):
    bearing = radians(bearing_deg)
    return {
        "id": actor_id,
        "x": distance * cos(bearing),
        "y": distance * sin(bearing),
        "yaw": 0.0,
        "v": v,
    }


def assert_detected(actor, sensor_name=None):
    wrapper = TrackLevelUncertaintyWrapper(seed=0)
    tracks, log = wrapper.update({"id": "ego", "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0}, [actor])
    assert log["num_observed_tracks"] == 1
    if sensor_name is not None:
        assert sensor_name in tracks[0].visible_sensor_names
    return tracks[0]


def main():
    ahead_track = assert_detected(actor_at("ahead_50", 50.0, 0.0), "front_long_radar")
    assert "front_camera" in ahead_track.visible_sensor_names

    wrapper = TrackLevelUncertaintyWrapper(seed=0)
    tracks, log = wrapper.update(
        {"id": "ego", "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0},
        [actor_at("ahead_200", 200.0, 0.0)],
    )
    assert tracks == []
    assert log["dropped_actor_ids"] == ["ahead_200"]
    assert log["drop_reasons"]["ahead_200"] == "out_of_range_or_fov"

    track_30 = assert_detected(actor_at("deg_30", 50.0, 30.0))
    assert (
        "front_near_radar" in track_30.visible_sensor_names
        or "front_left_corner_radar" in track_30.visible_sensor_names
    )

    rear_left_range = SENSOR_CONFIG["rear_left_corner_radar"]["range_m"]
    assert_detected(actor_at("deg_100", min(50.0, rear_left_range - 1.0), 100.0), "rear_left_corner_radar")
    assert_detected(actor_at("near_ultrasonic", 3.0, 165.0), "ultrasonic_11")

    ego = {"id": "ego", "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0}
    actor = actor_at("deterministic", 40.0, 5.0)
    track_a = TrackLevelUncertaintyWrapper(seed=42).update(ego, [actor])[0][0]
    track_b = TrackLevelUncertaintyWrapper(seed=42).update(ego, [actor])[0][0]
    track_c = TrackLevelUncertaintyWrapper(seed=43).update(ego, [actor])[0][0]

    assert np.isclose(track_a.x_rel, track_b.x_rel)
    assert np.isclose(track_a.y_rel, track_b.y_rel)
    assert np.isclose(track_a.v, track_b.v)
    assert not (
        np.isclose(track_a.x_rel, track_c.x_rel)
        and np.isclose(track_a.y_rel, track_c.y_rel)
        and np.isclose(track_a.v, track_c.v)
    )

    print("TrackLevelUncertaintyWrapper smoke test passed.")


if __name__ == "__main__":
    main()
