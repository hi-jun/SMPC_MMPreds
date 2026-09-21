"""One-off: re-place logged CARLA actors at a chosen instant and grab a top-down photo.

The scenario runner's drone camera only records live runs, so this replays poses from
scenario_result.pkl (rhs frame: y and yaw sign-flipped) into a static Town04 world.
"""
import json, pickle, sys, time, pathlib
import numpy as np
import carla
from PIL import Image

REPO = pathlib.Path("/home/core-dev/HJ/ACC/SMPC_MMPreds")
PROP = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000"
OUT = REPO / "overleaf" / "ppt"
# name, sweep, group, run, capture time relative to t0 [s]
SCENES = json.loads(sys.argv[1])
WIDTH_M, IMG_W, IMG_H, FOV = 70.0, 2400, 1000, 40.0
COLORS = {"ego": "0,90,210", "target_cutin": "210,30,30", "target_cutout": "210,30,30",
          "target_lead_after_cutout": "255,150,0"}
GRAY = "185,185,185"


def color_for(key):
    for prefix, c in COLORS.items():
        if key.startswith(prefix):
            return c
    return GRAY


def main():
    client = carla.Client("localhost", 2040)
    client.set_timeout(120.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town04"):
        world = client.load_world("Town04")
    world.set_weather(carla.WeatherParameters.ClearNoon)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    cmap = world.get_map()
    lib = world.get_blueprint_library()
    # Town04 has an overpass across this stretch (deck at z ~ 10 m over y 0..43); hide what is
    # elevated above the highway there so the top-down camera sees the cars under it.
    hide = set()
    for o in world.get_environment_objects(carla.CityObjectLabel.Any):
        b = o.bounding_box; c = b.location; e = b.extent
        if (c.z - e.z > 2.0 and c.y + e.y > -1.0 and c.y - e.y < 45.0 and c.x + e.x > -45.0 and c.x - e.x < 25.0
                and str(o.type) in ("Roads", "RoadLines", "Bridge", "GuardRail", "Walls", "Poles", "TrafficSigns")):
            hide.add(o.id)
    world.enable_environment_objects(hide, False)
    print("hid %d overpass objects" % len(hide))
    for name, sweep, group, run, t_rel in SCENES:
        d = REPO / "results" / "acc_scenario_sweep" / sweep / PROP / group / run
        data = pickle.load(open(d / "scenario_result.pkl", "rb"))
        ego = next(v for k, v in data.items() if k.startswith("ego"))
        ts = ego["policy_log"]["steps"][0]["time_s"] + t_rel
        actors, poses = [], []
        for key, val in data.items():
            if key.startswith("_"):
                continue
            tr = np.asarray(val["state_trajectory"], float)
            if tr.size == 0 or not (tr[0, 0] <= ts <= tr[-1, 0]):
                continue
            x = float(np.interp(ts, tr[:, 0], tr[:, 1]))
            y = -float(np.interp(ts, tr[:, 0], tr[:, 2]))
            yaw = -float(np.degrees(np.interp(ts, tr[:, 0], tr[:, 3])))
            wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
            z = wp.transform.location.z
            bp = lib.find("vehicle.audi.tt")
            if bp.has_attribute("color"):
                bp.set_attribute("color", color_for(key))
            a = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z + 3.0), carla.Rotation(yaw=yaw)))
            if a is None:
                print("spawn failed", name, key); continue
            a.set_simulate_physics(False)
            a.set_transform(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
            actors.append(a); poses.append((key, x, y, z))
        xs = np.array([p[1] for p in poses]); ys = np.array([p[2] for p in poses]); zs = np.array([p[3] for p in poses])
        cx, cy, cz = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()), float(np.median(zs))
        h = WIDTH_M / (2.0 * np.tan(np.radians(FOV / 2.0)))
        cam_bp = lib.find("sensor.camera.rgb")
        for k, v in (("image_size_x", IMG_W), ("image_size_y", IMG_H), ("fov", FOV)):
            cam_bp.set_attribute(k, str(v))
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=cx, y=cy, z=cz + h),
                                                        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))
        frames = []
        cam.listen(lambda img: frames.append(img))
        for _ in range(60):
            world.tick(); time.sleep(0.02)
        world.tick(); time.sleep(0.5)
        img = frames[-1]
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, 2::-1]
        out = OUT / ("bev_%s.png" % name)
        Image.fromarray(arr).save(out)
        print("wrote", out, "| actors:", ", ".join("%s@(%.1f,%.1f)" % (k, x, y) for k, x, y, _ in poses),
              "| camera centre (%.1f, %.1f), altitude %.0f m" % (cx, cy, h))
        cam.stop(); cam.destroy()
        for a in actors:
            a.destroy()
        world.tick()
    settings.synchronous_mode = False
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
