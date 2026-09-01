import os
import sys
from math import cos, radians, sin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge


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


def main():
    ego = {"id": "ego", "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0}
    rear_left_range = SENSOR_CONFIG["rear_left_corner_radar"]["range_m"]
    gt_actors = [
        actor_at("ahead_50m", 50.0, 0.0, 12.0),
        actor_at("ahead_200m_dropped", 200.0, 0.0, 12.0),
        actor_at("front_left_30deg", 50.0, 30.0, 8.0),
        actor_at("left_rear_100deg", min(50.0, rear_left_range - 1.0), 100.0, 6.0),
        actor_at("ultrasonic_close_165deg", 3.0, 165.0, 2.0),
        actor_at("gap_outside_fov", 90.0, 95.0, 6.0),
    ]

    wrapper = TrackLevelUncertaintyWrapper(noise_level="nominal", seed=7)
    tracks, log = wrapper.update(ego, gt_actors, timestamp=1.0)
    track_by_id = {track.id: track for track in tracks}

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_sensor_fovs(ax)

    ax.scatter([0.0], [0.0], s=120, c="black", marker="^", label="ego")
    ax.text(0.0, -6.0, "ego", ha="center", va="top")

    for actor in gt_actors:
        gt_x = actor["x"]
        gt_y = actor["y"]
        observed = actor["id"] in track_by_id
        color = "tab:green" if observed else "tab:red"
        marker = "o" if observed else "x"
        label = "GT observed" if observed else "GT dropped"
        ax.scatter([gt_x], [gt_y], s=80, c=color, marker=marker, label=label)
        ax.text(gt_x, gt_y + 4.0, actor["id"], ha="center", fontsize=9, color=color)

        if observed:
            track = track_by_id[actor["id"]]
            ax.scatter([track.x_rel], [track.y_rel], s=90, c="tab:orange", marker="X")
            ax.plot([gt_x, track.x_rel], [gt_y, track.y_rel], color="tab:orange", linewidth=1.2)
            sensor_text = ", ".join(track.visible_sensor_names[:2])
            if len(track.visible_sensor_names) > 2:
                sensor_text += ", ..."
            ax.text(
                track.x_rel,
                track.y_rel - 5.0,
                f"observed\n{sensor_text}",
                ha="center",
                va="top",
                fontsize=8,
                color="tab:orange",
            )

    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25.0, 210.0)
    ax.set_ylim(-90.0, 100.0)
    ax.set_xlabel("x_rel forward from ego [m]")
    ax.set_ylabel("y_rel left from ego [m]")
    ax.set_title("TrackLevelUncertaintyWrapper ego-view observed tracks")
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
            "track_level_uncertainty_visualization.png",
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


if __name__ == "__main__":
    main()
