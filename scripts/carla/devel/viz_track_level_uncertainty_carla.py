import os
import sys
from math import cos, radians, sin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(CARLA_SCRIPT_DIR)

import carla

from perception.track_level_uncertainty import (
    SENSOR_CONFIG,
    TrackLevelUncertaintyWrapper,
    transform_to_ego_frame,
)


def relative_to_carla_location(ego_tf, distance, bearing_deg, z_offset=0.0):
    yaw_rhs = -radians(float(ego_tf.rotation.yaw))
    bearing = radians(bearing_deg)
    x_rel = distance * cos(bearing)
    y_rel = distance * sin(bearing)
    ego_x_rhs = float(ego_tf.location.x)
    ego_y_rhs = -float(ego_tf.location.y)
    x_rhs = ego_x_rhs + cos(yaw_rhs) * x_rel - sin(yaw_rhs) * y_rel
    y_rhs = ego_y_rhs + sin(yaw_rhs) * x_rel + cos(yaw_rhs) * y_rel
    return carla.Location(x=x_rhs, y=-y_rhs, z=float(ego_tf.location.z) + z_offset)


def make_transform(ego_tf, distance, bearing_deg, yaw_deg=0.0, z_offset=0.0):
    location = relative_to_carla_location(ego_tf, distance, bearing_deg, z_offset=z_offset)
    return carla.Transform(location, carla.Rotation(yaw=yaw_deg))


def draw_sensor_fovs(ax):
    for sensor_name, sensor in SENSOR_CONFIG.items():
        if not sensor.get("enabled", True):
            continue
        center = sensor["angle_center_deg"]
        half = sensor["half_angle_deg"]
        radius = sensor["range_m"]
        alpha = 0.025 if sensor_name.startswith("ultrasonic") else 0.055
        wedge = Wedge(
            (0.0, 0.0),
            radius,
            center - half,
            center + half,
            facecolor="tab:blue",
            edgecolor="none",
            alpha=alpha,
            linewidth=0.0,
        )
        ax.add_patch(wedge)


def spawn_vehicle(world, blueprint, transform):
    actor = world.spawn_actor(blueprint, transform)
    actor.set_simulate_physics(False)
    return actor


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    world.wait_for_tick()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("role_name", "track_wrapper_validation")

    actors = []
    try:
        base_tf = world.get_map().get_spawn_points()[0]
        ego_tf = carla.Transform(
            carla.Location(x=base_tf.location.x, y=base_tf.location.y, z=base_tf.location.z + 80.0),
            carla.Rotation(yaw=0.0),
        )
        ego_actor = spawn_vehicle(world, vehicle_bp, ego_tf)
        actors.append(ego_actor)

        rear_left_range = SENSOR_CONFIG["rear_left_corner_radar"]["range_m"]
        actor_specs = [
            ("ahead_50m", 50.0, 0.0),
            ("ahead_200m_dropped", 200.0, 0.0),
            ("front_left_30deg", 50.0, 30.0),
            ("left_rear_100deg", min(50.0, rear_left_range - 1.0), 100.0),
            ("ultrasonic_close_165deg", 3.0, 165.0),
            ("gap_outside_fov", 90.0, 95.0),
        ]

        surrounding_actors = []
        for idx, (name, distance, bearing_deg) in enumerate(actor_specs):
            actor_bp = blueprint_library.find("vehicle.tesla.model3")
            actor_bp.set_attribute("role_name", name)
            tf = make_transform(ego_tf, distance, bearing_deg, yaw_deg=0.0, z_offset=10.0 * (idx + 1))
            actor = spawn_vehicle(world, actor_bp, tf)
            actors.append(actor)
            surrounding_actors.append(actor)

        world.wait_for_tick()

        wrapper = TrackLevelUncertaintyWrapper(noise_level="nominal", seed=11)
        tracks, log = wrapper.update(ego_actor, surrounding_actors, timestamp=world.get_snapshot().timestamp.elapsed_seconds)
        track_by_id = {track.id: track for track in tracks}

        fig, ax = plt.subplots(figsize=(12, 8))
        draw_sensor_fovs(ax)
        ax.scatter([0.0], [0.0], s=120, c="black", marker="^", label="ego actor")
        ax.text(0.0, -6.0, "ego actor", ha="center", va="top")

        for actor, (name, _, _) in zip(surrounding_actors, actor_specs):
            x_rel, y_rel, _, _ = transform_to_ego_frame(ego_actor, actor)
            observed = actor.id in track_by_id
            color = "tab:green" if observed else "tab:red"
            marker = "o" if observed else "x"
            label = "CARLA GT observed" if observed else "CARLA GT dropped"
            ax.scatter([x_rel], [y_rel], s=80, c=color, marker=marker, label=label)
            ax.text(x_rel, y_rel + 4.0, name, ha="center", fontsize=9, color=color)

            if observed:
                track = track_by_id[actor.id]
                ax.scatter([track.x_rel], [track.y_rel], s=90, c="tab:orange", marker="X")
                ax.plot([x_rel, track.x_rel], [y_rel, track.y_rel], color="tab:orange", linewidth=1.2)

        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.axvline(0.0, color="0.4", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-25.0, 210.0)
        ax.set_ylim(-90.0, 100.0)
        ax.set_xlabel("x_rel forward from ego actor [m]")
        ax.set_ylabel("y_rel left from ego actor [m]")
        ax.set_title("CARLA actor validation: TrackLevelUncertaintyWrapper observed tracks")
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="upper right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "results",
                "track_level_uncertainty_carla_validation.png",
            )
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

        print(f"saved_png={output_path}")
        print(f"num_gt_actors={log['num_gt_actors']}")
        print(f"num_observed_tracks={log['num_observed_tracks']}")
        print(f"num_dropped_tracks={log['num_dropped_tracks']}")
        print(f"dropped_actor_ids={log['dropped_actor_ids']}")
        print("observed_track_ids=" + str([track.id for track in tracks]))
        print("actor_name_by_id=" + str({actor.id: name for actor, (name, _, _) in zip(surrounding_actors, actor_specs)}))
    finally:
        for actor in actors:
            if actor.is_alive:
                actor.destroy()


if __name__ == "__main__":
    main()
