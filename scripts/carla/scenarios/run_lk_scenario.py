from dataclasses import dataclass
from typing import List

import carla
import os
import sys
import pdb
import cv2
import numpy as np
import random
import pickle
import time

_REUSED_CARLA_WORLD_KEYS = set()


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "/PythonAPI"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)

from policies.static_agent import StaticAgent
from policies.smpc_agent import SMPCAgent
from policies.mpc_agent import MPCAgent
from policies.bl_smpc_agent import BLSMPCAgent
from policies.distance_triggered_lane_change_agent import DistanceTriggeredLaneChangeAgent
from policies.distance_triggered_cutout_agent import DistanceTriggeredCutOutAgent
from policies.fixed_lane_speed_agent import FixedLaneSpeedAgent
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent

from rasterizer.agent_history import AgentHistory
from rasterizer.sem_box_rasterizer import SemBoxRasterizer
from utils.frenet_trajectory_handler import fix_angle
from utils.vehicle_geometry_utils import vehicle_name_to_lf_lr

rootdir = os.path.abspath(__file__).split('scripts')[0]
sys.path.append(rootdir)
scriptdir = rootdir + 'scripts/'
sys.path.append(scriptdir)

"""
Simulation parameter classes.
"""
@dataclass(frozen=True)
class CarlaParams:
    # Carla world settings + intersection definition.
    map_str               : str # e.g. "Town05"
    weather_str           : str # e.g. "ClearNoon"
    fps                   : int # what fps to run the simulator in synchronous mode
    intersection_csv_loc  : str # file location of the csv defining the intersection

    # Carla client settings.
    ip_addr        : str   = "localhost"
    port           : int   = 2000
    timeout_period : float = 999
    max_sim_time_s : float = 30.0
    spawn_settle_s : float = 0.0
    spawn_settle_realtime : bool = False
    cruise_warmup_s : float = 0.0

@dataclass(frozen=True)
class DroneVizParams:
    # Parameters for the "drone": camera used to capture Carla scene.
    # By default, this represents a top down view.
    # XY must be specified, as it varies with the choice of intersection.
    x          : float
    z          : float
    y          : float =   0.
    roll       : float =   0.
    pitch      : float =  -90.
    yaw        : float =   0.
    img_width  : int   = 1920
    img_height : int   = 1080
    fov        : int   = 90

    # Parameters for how to handle OpenCV img corresponding to the drone.
    visualize_opencv      : bool = True # show OpenCV window as the simulation is occuring
    save_avi              : bool = True # whether to save OpenCV visualization as a video
    overlay_gmm           : bool = True # whether to show the confidence ellipses for predicted agents.
    overlay_ego_info      : bool = True # add a string with text about ego's state/control.
    overlay_mode_probs    : bool = True # add a string with the mode probabilities.
    overlay_traj_hist     : bool = True # add the trajectory history for each agent

@dataclass(frozen=True)
class VehicleParams:
    # High level vehicle/policy selection.
    role          : str # "ego", "target", "traffic" [dynamic neighbor], or "static"
    vehicle_type  : str # currently use one of {"vehicle.audi.tt", "vehicle.mercedes-benz.coupe"}
    vehicle_color : str # currently use "246, 246, 246" for static, "186, 0, 0" for ego, and "65, 63, 197" for dynamic
    policy_type   : str # {"static", mpc", "smpc", "blsmpc", "distance_triggered_lane_change", "fixed_lane_speed"} -> which control policy to use for this agent

    # Initial state and goal location selection.
    intersection_start_node_idx : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
    intersection_goal_node_idx  : int        # {0, 1, 2, 3} -> corresponds to a direction in the intersection_json above
    start_left_offset           : float      # how far to move the car's start pose in its local left (i.e. lateral axis) direction (m)
    goal_left_offset            : float      # how far to move the car's goal pose in its local left (i.e. lateral axis) direction (m)
    start_longitudinal_offset   : float      # how far to move the car's start pose in its local fwd (i.e. longitudinal axis) direction (m)
    goal_longitudinal_offset    : float      # how far to move the car's goal pose in its local fwd (i.e. longitudinal axis) direction (m)
    nominal_speed               : float      # how fast the car should travel if unobstructed / not turning
    init_speed                  : float      # the car's initial speed in simulation (m/s)

    # General MPC parameters.  Some of these can be ignored (e.g. n_modes if using MPCAgent).
    N         : int   = 10  # horizon of the MPC solution
    dt        : float = 0.2 # timestep of the discretization used (s)
    num_modes : int   = 2   # number of GMM modes considered by MPC (prioritizing most probable ones first)

    # SMPC specific parameters (ignored for any other policy_type).
    smpc_config : str = "full" # "full", "open_loop", "no_switch"

    # Distance-triggered lane change parameters.
    lane_change_trigger_distance : float = 30.0
    lane_change_trigger_mode : str = "ego_gap"   # or "lead_gap": distance to the lead in the vehicle's own lane
    lane_change_distance_same_lane : float = 5.0
    lane_change_distance_other_lane : float = 100.0
    lane_change_distance : float = 25.0
    cutout_direction : str = "left"
    # Speed cap toward the same-lane lead (distance_triggered_lane_change);
    # defaults mirror SPEED_CAP_MIN_LONGITUDINAL_GAP_M / SPEED_CAP_GAIN.
    speed_cap_min_gap_m : float = 2.5
    speed_cap_gain : float = 1.2

@dataclass(frozen=True)
class PredictionParams:
    predictor_type       : str = "stdan"

    # Model parameter locations, given relative to <ROOTDIR>/scripts/models/
    model_weights         : str = "l5kit_multipath_10/"
    model_anchors         : str = "l5kit_clusters_16.npy"

    # STDAN settings, with checkpoint path relative to <ROOTDIR>.
    stdan_ckpt            : str = "predictor/stdan/ckpt/best_model.pt"
    stdan_history         : float = 3.0
    stdan_future          : float = 5.0
    stdan_dt              : float = 0.1
    stdan_input_coordinates : str = "carla"
    stdan_mc_dropout      : bool = False

    # Flag to render traffic lights on rasterized image for prediction.
    render_traffic_lights : bool = False # set by default to false since agents ignore lights at the moment.

    # TODO: future work includes things like how often to update preds (if not at the Carla fps).

"""
Util functions for Carla. # TODO: move this elsewhere.
"""
def load_intersection(intersection_csv):
    with open(intersection_csv, 'r') as f:
        lines = f.readlines()

    intersection = []

    for line in lines:
        if '#' in line or len(line)==0:
            continue # comment
        data = line.replace(" ", "").replace("\n","").split(",")

        start_pose = [float(data[0]), float(data[1]), int(data[2])]
        goal_pose  = [float(data[3]), float(data[4]), int(data[5])]
        intersection.append( [start_pose, goal_pose] )

    return intersection

def get_vehicle_policy(vehicle_params, vehicle_actor, goal_transform):
    if vehicle_params.policy_type == "static":
        return StaticAgent(vehicle_actor, goal_transform.location, nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "mpc":
        return MPCAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "distance_triggered_lane_change":
        return DistanceTriggeredLaneChangeAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed,
                        trigger_distance_m=vehicle_params.lane_change_trigger_distance,
                        trigger_mode=vehicle_params.lane_change_trigger_mode,
                        distance_same_lane=vehicle_params.lane_change_distance_same_lane,
                        distance_other_lane=vehicle_params.lane_change_distance_other_lane,
                        distance_lane_change=vehicle_params.lane_change_distance,
                        speed_cap_min_gap_m=vehicle_params.speed_cap_min_gap_m,
                        speed_cap_gain=vehicle_params.speed_cap_gain)
    elif vehicle_params.policy_type == "distance_triggered_cutout":
        return DistanceTriggeredCutOutAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed,
                        trigger_distance_m=vehicle_params.lane_change_trigger_distance,
                        distance_same_lane=vehicle_params.lane_change_distance_same_lane,
                        distance_other_lane=vehicle_params.lane_change_distance_other_lane,
                        distance_lane_change=vehicle_params.lane_change_distance,
                        cutout_direction=vehicle_params.cutout_direction)
    elif vehicle_params.policy_type == "fixed_lane_speed":
        return FixedLaneSpeedAgent(vehicle_actor, goal_transform.location, \
                        nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "acc_nair_smpc":
        return ACCNairSMPCAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed,
                        smpc_config=vehicle_params.smpc_config)
    elif vehicle_params.policy_type == "blsmpc":
        return BLSMPCAgent(vehicle_actor, goal_transform.location, \
                        N=vehicle_params.N,
                        dt=vehicle_params.dt,
                        N_modes=vehicle_params.num_modes,
                        nominal_speed_mps=vehicle_params.nominal_speed)
    elif vehicle_params.policy_type == "smpc":
        if vehicle_params.smpc_config.endswith("OAinner"):
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config.split("_OAinner")[0],
                            OAIA=True)
        elif vehicle_params.smpc_config.endswith("obca"):
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config.split("_obca")[0][:-2],
                            obca=True,
                            obca_mode=int(vehicle_params.smpc_config.split("_obca")[0][-1]))
        else :
            return SMPCAgent(vehicle_actor, goal_transform.location, \
                            N=vehicle_params.N,
                            dt=vehicle_params.dt,
                            N_modes=vehicle_params.num_modes,
                            nominal_speed_mps=vehicle_params.nominal_speed,
                            smpc_config=vehicle_params.smpc_config)
    else:
        raise ValueError(f"Unsupported policy type: {vehicle_params.policy_type}")

def get_intersection_transform(intersection, vehicle_params, endpoint_str, spawn_height=2.0):
    node_idx            = None
    endpoint_idx        = None
    left_offset         = None
    longitudinal_offset = None

    if endpoint_str == "start":
        endpoint_idx        = 0
        node_idx            = vehicle_params.intersection_start_node_idx
        left_offset         = vehicle_params.start_left_offset
        longitudinal_offset = vehicle_params.start_longitudinal_offset

    elif endpoint_str == "goal":
        endpoint_idx        = 1
        node_idx            = vehicle_params.intersection_goal_node_idx
        left_offset         = vehicle_params.goal_left_offset
        longitudinal_offset = vehicle_params.goal_longitudinal_offset
    else:
        raise ValueError(f"Invalid endpoint:{endpoint_str}.  Expected start or goal.")

    # Extract the pose from the intersection definition.
    x, y, yaw_deg = intersection[node_idx][endpoint_idx]
    yaw_rad = np.radians(float(yaw_deg))

    # Translate the pose given the longitudinal offset.
    x+=longitudinal_offset*np.cos(yaw_rad)
    y+=longitudinal_offset*np.sin(yaw_rad)
    # Translate the pose given the lateral/left offset.
    left_dir_yaw = yaw_rad - np.pi/2.
    x += left_offset * np.cos( left_dir_yaw )
    y += left_offset * np.sin( left_dir_yaw )

    # Make the Carla transform.
    loc = carla.Location(x = x, y = y, z = spawn_height)
    rot = carla.Rotation(yaw=yaw_deg)
    # import pdb; pdb.set_trace()
    return carla.Transform(loc, rot)

def transform_to_local_frame(motion_hist_array):
    # TODO: clean up / document / move.
    local_x, local_y, local_yaw = motion_hist_array[-1, 1:]

    R_local_to_world    = np.array([[np.cos(local_yaw), -np.sin(local_yaw)],\
                                    [np.sin(local_yaw),  np.cos(local_yaw)]])
    t_local_to_world    = np.array([local_x, local_y])

    R_world_to_local =  R_local_to_world.T
    t_world_to_local = -R_local_to_world.T @ t_local_to_world

    for t in range(motion_hist_array.shape[0]):
        xy_global = motion_hist_array[t, 1:3]
        xy_local  = R_world_to_local @ xy_global + t_world_to_local
        motion_hist_array[t, 1:3] = xy_local

        pose_diff = motion_hist_array[t, 3] - local_yaw

        if ~np.isnan(pose_diff):
            motion_hist_array[t, 3] = fix_angle(pose_diff)

    return motion_hist_array, R_local_to_world, t_local_to_world

def get_target_agent_history(agent_history, target_agent_id):
    # TODO: clean up / document / move.
    snapshot = agent_history.query(history_secs = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    tms   = []
    poses = []

    for k in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        tms.append(k)
        snapshot_key = np.round(k, 2)
        if(len(snapshot[snapshot_key]) == 0):
            poses.append([None, None, None])
        else:
            for entry in snapshot[snapshot_key]['vehicles']:
                if entry['id'] == target_agent_id:
                    pose = entry['centroid']
                    pose.append(entry['yaw'])
                    poses.append( pose )
                    break
    tms = [-v if v > 0. else 0. for v in tms]
    motion_hist_array = np.column_stack((tms, poses)).astype(np.float32)

    return transform_to_local_frame(motion_hist_array)

"""
Main class to simulate and run parametrized scenarios.
"""
class RunLKScenario:
    def __init__(self,
                 carla_params        : CarlaParams,
                 drone_viz_params    : DroneVizParams,
                 vehicle_params_list : List[VehicleParams],
                 prediction_params   : PredictionParams,
                 savedir : str):
        # Prediction setup uses the simulator FPS to size the history buffer.
        self.timeout   = carla_params.timeout_period
        self.carla_fps = carla_params.fps
        self.max_iters = int(self.carla_fps*carla_params.max_sim_time_s)
        self.spawn_settle_s = max(0.0, float(carla_params.spawn_settle_s))
        self.spawn_settle_realtime = bool(carla_params.spawn_settle_realtime)
        self.spawn_settle_log = {}
        self.cruise_warmup_s = max(0.0, float(carla_params.cruise_warmup_s))
        self.cruise_warmup_log = {}
        try:
            self._setup_carla_world(carla_params)
            self._setup_vehicles(vehicle_params_list, carla_params)
            self.drone = None
            if drone_viz_params.save_avi or drone_viz_params.visualize_opencv:
                self._setup_camera(drone_viz_params)
            self.use_prediction_model = self._should_use_prediction_model(
                drone_viz_params, vehicle_params_list)
            if self.use_prediction_model:
                self._setup_predictions(prediction_params)
        except Exception as e:
            print("Failed to setup the scenario!")
            raise e

        # For logging results + videos.
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

        # Needed for Sync mode loop.

        # Needed for OpenCV/Carla world visualization.
        self.viz_params = drone_viz_params
        # self.mode_rgb_colors = [(255, 0, 255), (255, 255, 0)]#, (0, 255, 255)] # TODO: autogenerate
        self.mode_rgb_colors = [
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
            (255, 128, 0),
            (0, 200, 80),
            (80, 120, 255),
            (200, 200, 200),
            (255, 80, 120),
            (120, 255, 80),
        ]

    def _tick_vehicles_stationary(self, sync_mode, duration_s, realtime):
        delay_ticks = int(np.ceil(float(duration_s) * self.carla_fps))
        zero_vel = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        hold_control = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        tick_dt = 1.0 / float(self.carla_fps)
        wall_start = time.monotonic()
        sim_start = None
        sim_end = None
        for _ in range(delay_ticks):
            tick_start = time.monotonic()
            for veh_actor in self.vehicle_actors:
                veh_actor.set_target_velocity(zero_vel)
                veh_actor.apply_control(hold_control)
            sync_data = sync_mode.tick(timeout=self.timeout)
            snap = sync_data[0]
            elapsed_seconds = getattr(snap, "elapsed_seconds", snap.timestamp.elapsed_seconds)
            if sim_start is None:
                sim_start = elapsed_seconds
            sim_end = elapsed_seconds
            if realtime:
                time.sleep(max(0.0, tick_dt - (time.monotonic() - tick_start)))

        release_control = carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
        for veh_actor in self.vehicle_actors:
            veh_actor.set_target_velocity(zero_vel)
            veh_actor.apply_control(release_control)
        return {
            "requested_s": float(duration_s),
            "ticks": int(delay_ticks),
            "realtime": bool(realtime),
            "wall_elapsed_s": float(time.monotonic() - wall_start),
            "sim_elapsed_s": float(0.0 if sim_start is None or sim_end is None else sim_end - sim_start),
        }

    def _tick_vehicles_cruising(self, sync_mode, duration_s):
        # Hold every vehicle at its nominal speed so the predictor history fills
        # with steady-state motion. Without this the history window covers the
        # spawn transient -- vehicles sag to ~55% of their set speed and
        # accelerate back -- which STDAN reads as lane-change intent.
        warmup_ticks = int(np.ceil(float(duration_s) * self.carla_fps))
        history_updaters = [
            getattr(policy, "_update_agent_history")
            for policy in self.vehicle_policies
            if callable(getattr(policy, "_update_agent_history", None))
        ]
        speeds = []
        sim_start = None
        sim_end = None
        for _ in range(warmup_ticks):
            # Re-apply the target velocity every tick. A single injection decays
            # immediately because no throttle is being commanded yet.
            for veh_actor, init_speed in zip(self.vehicle_actors, self.vehicle_init_speeds):
                yaw_carla = veh_actor.get_transform().rotation.yaw
                veh_actor.set_target_velocity(carla.Vector3D(
                    x=init_speed*np.cos(np.radians(yaw_carla)),
                    y=init_speed*np.sin(np.radians(yaw_carla)),
                    z=0.))
            # Drive the low-level controllers too. They low-pass the throttle
            # against a control_prev that starts at zero, so handing over from
            # set_target_velocity with a cold filter drops the car ~6 m/s before
            # the throttle catches up -- which would eat several seconds of the
            # approach phase. Running them here warms the filter and the
            # drivetrain while set_target_velocity still pins the speed.
            for vehicle_actor, policy, nominal in zip(
                    self.vehicle_actors, self.vehicle_policies, self.vehicle_init_speeds):
                low_level = getattr(policy, "_low_level_control", None)
                if low_level is None:
                    continue
                velocity = vehicle_actor.get_velocity()
                speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
                vehicle_actor.apply_control(
                    low_level.update(speed, 0.0, float(nominal), 0.0))
            sync_data = sync_mode.tick(timeout=self.timeout)
            snap = sync_data[0]
            elapsed_seconds = getattr(snap, "elapsed_seconds", snap.timestamp.elapsed_seconds)
            if sim_start is None:
                sim_start = elapsed_seconds
            sim_end = elapsed_seconds
            # Fill the prediction histories without running any policy, so the
            # lane-change trigger stays armed and nothing is logged yet.
            if self.use_prediction_model:
                self.agent_history.update(snap, self.world)
            for update_history in history_updaters:
                update_history()
            ego_vel = self.vehicle_actors[self.ego_vehicle_idx].get_velocity()
            speeds.append(float(np.linalg.norm([ego_vel.x, ego_vel.y])))
        return {
            "requested_s": float(duration_s),
            "ticks": int(warmup_ticks),
            "sim_elapsed_s": float(0.0 if sim_start is None or sim_end is None else sim_end - sim_start),
            "history_updaters": len(history_updaters),
            "ego_speed_min": float(min(speeds)) if speeds else None,
            "ego_speed_max": float(max(speeds)) if speeds else None,
            "ego_speed_final": float(speeds[-1]) if speeds else None,
        }

    def run_scenario(self):
        # Return flag to indicate if this ran to completion.
        ran_successfully = False

        # Video Setup
        writer = None
        if self.viz_params.save_avi:
            avi_name = os.path.join(self.savedir, "carla_sim.avi")
            writer   = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc(*'MJPG'), self.carla_fps, (self.viz_params.img_width, self.viz_params.img_height))

        # Data Logging Setup
        self.results_dict = {}
        self.prediction_log = []
        for ind_vehicle, vehicle in enumerate(self.vehicle_actors):
            key = self.vehicle_actor_keys[ind_vehicle]
            l_f, l_r = vehicle_name_to_lf_lr(vehicle.type_id) # e.g. "vehicle.audi.tt"
            self.results_dict[key] = {"l_f"              : l_f,
                                      "l_r"              : l_r,
                                      "state_trajectory" : [],
                                      "lane_trajectory"  : [],
                                      "input_trajectory" : [],
                                      "feasibility"      : [],
                                      "solve_times"      : []}

        try:
            sensors = [self.drone] if self.drone is not None else []
            with CarlaSyncMode(self.world, *sensors, fps=self.carla_fps) as sync_mode:
                # Fast-forward spawn settling before scenario control/logging begins.
                self.spawn_settle_log = self._tick_vehicles_stationary(
                    sync_mode,
                    self.spawn_settle_s,
                    self.spawn_settle_realtime,
                )

                # Set initial velocity for all vehicle agents.
                for veh_actor, init_speed in zip(self.vehicle_actors, self.vehicle_init_speeds):
                    yaw_carla = veh_actor.get_transform().rotation.yaw
                    carla_vel = carla.Vector3D(x=init_speed*np.cos(np.radians(yaw_carla)) ,
                                               y=init_speed*np.sin(np.radians(yaw_carla)) ,
                                               z=0.)
                    veh_actor.set_target_velocity(carla_vel)

                for _ in range(1):
                    sync_mode.tick(timeout=self.timeout)

                # Steady-state cruise so the predictor history holds real motion
                # rather than the spawn transient before control/logging begins.
                if self.cruise_warmup_s > 0.0:
                    self.cruise_warmup_log = self._tick_vehicles_cruising(
                        sync_mode,
                        self.cruise_warmup_s,
                    )

                # Loop until all vehicles have reached their goal or we've exceeded self.max_iters.
                for _ in range(self.max_iters):
                    sync_data = sync_mode.tick(timeout=self.timeout)
                    snap = sync_data[0]
                    img = sync_data[1] if self.drone is not None else None

                    # Handle predictions.
                    if self.use_prediction_model:
                        self.agent_history.update(snap, self.world)
                        tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred = self._make_predictions()
                    else:
                        tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred = [], [], [], []
                    pred_dict={ "tvs_positions": tvs_positions, "tvs_mode_dists": tvs_mode_dists}

                    # Run policies for each agent.
                    t_elapsed = getattr(snap, "elapsed_seconds", snap.timestamp.elapsed_seconds)
                    prediction_debug = getattr(self, "_last_prediction_debug", None)
                    if prediction_debug is not None:
                        prediction_debug["time"] = t_elapsed
                        self.prediction_log.append(prediction_debug)
                        self._last_prediction_debug = None
                    completed = True

                    if self.drone is not None:
                        ev_loc=self.vehicle_actors[self.ego_vehicle_idx].get_location()
                        cam_loc=carla.Location(x=ev_loc.x,y=ev_loc.y,z=ev_loc.z+self.drone_z_offset)
                        self.drone.set_location(cam_loc)



                    for idx_act, (act, policy) in enumerate(zip(self.vehicle_actors, self.vehicle_policies)):
                        if not self.vehicle_active[idx_act]:
                            continue
                        control, z0, u0, is_feasible, solve_time = policy.run_step(pred_dict)
                        if not policy.done():
                            z0 = np.append(t_elapsed, z0) # add the Carla timestamp
                            act_key = self.vehicle_actor_keys[idx_act]
                            self.results_dict[act_key]["state_trajectory"].append(z0)
                            self.results_dict[act_key]["lane_trajectory"].append(
                                self._actor_lane_record(act, t_elapsed))
                            self.results_dict[act_key]["input_trajectory"].append(u0)
                            self.results_dict[act_key]["feasibility"].append(is_feasible)
                            self.results_dict[act_key]["solve_times"].append(solve_time)

                        # true at the end of the loop only if all agents are done or if iter_ctr>=max_iters
                        if idx_act != self.ego_vehicle_idx and policy.done():
                            self._retire_done_actor(idx_act, t_elapsed, "policy_done")
                            continue
                        act.apply_control(control)

                        if idx_act == self.ego_vehicle_idx:
                            # Keep track of ego's information for rendering.
                            ego_vel   = act.get_velocity()
                            ego_speed = np.linalg.norm([ego_vel.x, ego_vel.y])
                            ego_ctrl  = control
                            completed = completed and policy.done()

                    if self.drone is not None:
                        # Get drone camera image.
                        img_drone = np.frombuffer(img.raw_data, dtype=np.uint8)
                        img_drone = np.reshape(img_drone, (img.height, img.width, 4))
                        img_drone = img_drone[:, :, :3]
                        img_drone = cv2.resize(img_drone, (self.viz_params.img_width, self.viz_params.img_height), interpolation = cv2.INTER_AREA)

                        # Handle overlays on drone camera image.
                        if self.viz_params.overlay_ego_info:
                            ego_str = f"EGO - v:{ego_speed:.3f}, th: {ego_ctrl.throttle:.2f}, bk: {ego_ctrl.brake:.2f}, st: {ego_ctrl.steer:.2f}"
                            cv2.putText(img_drone, ego_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if self.viz_params.overlay_gmm:
                            # if tvs_valid_pred[0]: # TODO: generalize this to multiple TVs.
                            self._viz_gmm(img_drone, tvs_mode_dists)

                        if self.viz_params.overlay_traj_hist:
                            self._viz_traj_hist(img_drone)

                        if self.viz_params.overlay_mode_probs:
                            if tvs_valid_pred and tvs_valid_pred[0]: # TODO: generalize this to multiple TVs.
                                cv2.putText(img_drone, "Mode probabilities: ", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                for prob_idx, mode_prob in enumerate(tvs_mode_probs[0]):
                                    cv2.putText(img_drone, f"{mode_prob:.3f}",
                                                (360 + prob_idx * 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, self.mode_rgb_colors[prob_idx % len(self.mode_rgb_colors)], 2)

                        # Handle visualization / saving to video.
                        if self.viz_params.visualize_opencv:
                            cv2.imshow("Drone", img_drone); cv2.waitKey(1)

                        if self.viz_params.save_avi:
                            writer.write(img_drone)

                    if completed:
                        # All cars reached their destinations, end before self.max_iters.
                        break

                # Save results and mark successful completion.
                for act_key in self.results_dict:
                    for arr_key in ["state_trajectory",
                                    "input_trajectory",
                                    "feasibility",
                                    "solve_times"]:
                        self.results_dict[act_key][arr_key] = np.array(self.results_dict[act_key][arr_key])
                for idx_act, (act, policy) in enumerate(zip(self.vehicle_actors, self.vehicle_policies)):
                    act_key = self.vehicle_actor_keys[idx_act]
                    self.results_dict[act_key]["collision_events"] = [
                        event for event in self.collision_events
                        if event.get("actor_index") == idx_act
                    ]
                    if hasattr(policy, "get_cut_in_log"):
                        self.results_dict[act_key]["policy_log"] = policy.get_cut_in_log()
                self.results_dict["_prediction_log"] = self.prediction_log
                self.results_dict["_spawn_settle_log"] = self.spawn_settle_log
                self.results_dict["_cruise_warmup_log"] = self.cruise_warmup_log
                self.results_dict["_collision_log"] = self.collision_events
                self.results_dict["_collision_count"] = len(self.collision_events)
                self.results_dict["_collision_sensor_errors"] = self.collision_sensor_errors
                self.results_dict["_retired_actors"] = self.retired_actor_records
                pkl_name = os.path.join(self.savedir, "scenario_result.pkl")
                pickle.dump(self.results_dict, open(pkl_name, "wb"))
                ran_successfully = True

        # Teardown.
        finally:
            if writer:
                writer.release()
            for sensor in getattr(self, "collision_sensors", []):
                if sensor is not None:
                    try:
                        sensor.stop()
                    except RuntimeError:
                        pass
                    self._destroy_actor_quietly(sensor)
            for actor in self.vehicle_actors:
                self._destroy_actor_quietly(actor)
            if self.drone is not None:
                self._destroy_actor_quietly(self.drone)
            cv2.destroyAllWindows()

        return ran_successfully

    def _actor_lane_record(self, actor, time_s):
        record = {
            "time_s": float(time_s),
            "road_id": None,
            "section_id": None,
            "lane_id": None,
            "lane_type": None,
            "waypoint_x": None,
            "waypoint_y": None,
            "waypoint_yaw": None,
        }
        try:
            waypoint = self.world.get_map().get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
        except Exception:
            waypoint = None
        if waypoint is None:
            return record
        loc = waypoint.transform.location
        record.update({
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "lane_id": waypoint.lane_id,
            "lane_type": str(waypoint.lane_type),
            "waypoint_x": float(loc.x),
            "waypoint_y": float(loc.y),
            "waypoint_yaw": float(waypoint.transform.rotation.yaw),
        })
        return record

    @staticmethod
    def _should_use_prediction_model(drone_viz_params, vehicle_params_list):
        needs_prediction_policy = any(
            vehicle.policy_type in ["smpc", "blsmpc"]
            for vehicle in vehicle_params_list)
        needs_prediction_overlay = (
            drone_viz_params.save_avi or drone_viz_params.visualize_opencv) and (
            drone_viz_params.overlay_gmm or drone_viz_params.overlay_mode_probs)
        return needs_prediction_policy or needs_prediction_overlay

    def _make_predictions(self):
        if self.prediction_model_type == "stdan":
            return self._make_stdan_predictions()
        return self._make_multipath_predictions()

    def _fake_target_prediction(self):
        ego_location = self.vehicle_actors[self.ego_vehicle_idx].get_location()
        ego_x, ego_y = ego_location.x, -ego_location.y
        curr_target_vehicle_position = [1000 + ego_x, 1000 + ego_y]
        tvs_positions = [curr_target_vehicle_position]
        tvs_mode_probs = [np.ones(self.ego_num_modes) / self.ego_num_modes]
        tvs_mode_dists = [
            [np.stack([[curr_target_vehicle_position] * self.ego_N] * self.ego_num_modes)],
            [np.stack([[np.identity(2)] * self.ego_N] * self.ego_num_modes)],
        ]
        tvs_valid_pred = [False]
        return tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred

    def _actor_state_rhs(self, actor):
        loc = actor.get_location()
        yaw = -fix_angle(np.radians(actor.get_transform().rotation.yaw))
        return np.array([loc.x, -loc.y, yaw], dtype=np.float64)

    def _actor_position_rhs(self, actor):
        state = self._actor_state_rhs(actor)
        return state[:2]

    def _constant_velocity_prediction(self, actor_idx, curr_position):
        actor = self.vehicle_actors[actor_idx]
        policy = self.vehicle_policies[actor_idx]
        vel = actor.get_velocity()
        velocity_rhs = np.array([vel.x, -vel.y], dtype=np.float64)
        if np.linalg.norm(velocity_rhs) < 1e-3:
            yaw = self._actor_state_rhs(actor)[2]
            nominal_speed = getattr(policy, "nominal_speed", 0.0)
            velocity_rhs = nominal_speed * np.array([np.cos(yaw), np.sin(yaw)])

        times = np.arange(1, self.ego_N + 1, dtype=np.float64) * getattr(policy, "DT", 0.2)
        mean_traj = curr_position + times[:, None] * velocity_rhs[None, :]
        mus = np.stack([mean_traj] * self.ego_num_modes)
        sigmas = np.stack([[0.8 * np.identity(2)] * self.ego_N for _ in range(self.ego_num_modes)])
        probs = np.ones(self.ego_num_modes) / self.ego_num_modes
        return probs, mus, sigmas

    def _selected_prediction_actor_idxs(self):
        active_tv_idxs = [
            idx for idx in self.tv_vehicle_idxs
            if idx < len(self.vehicle_active) and self.vehicle_active[idx]
        ]
        if len(active_tv_idxs) == 0:
            return []
        # SMPC_MMPreds is instantiated with a fixed TV problem bank.  Do not
        # fabricate duplicate target slots beyond what the ego policy supports.
        ego_policy = self.vehicle_policies[self.ego_vehicle_idx]
        max_supported_tvs = len(active_tv_idxs)
        if hasattr(ego_policy, "SMPC") and hasattr(ego_policy.SMPC, "N_TV_max"):
            max_supported_tvs = ego_policy.SMPC.N_TV_max
        elif hasattr(ego_policy, "NUM_TVS"):
            max_supported_tvs = ego_policy.NUM_TVS

        selected = [active_tv_idxs[0]]
        if len(active_tv_idxs) > 1:
            selected.append(active_tv_idxs[-1])
        return selected[:max_supported_tvs]

    def _stdan_trackings(self):
        snapshots = self.agent_history.query(
            history_secs=self.stdan_history_secs,
            closeness_eps=self.stdan_history_closeness_eps,
        )
        trackings = {}
        for idx, actor in enumerate(self.vehicle_actors):
            if idx < len(self.vehicle_active) and not self.vehicle_active[idx]:
                continue
            history = []
            for hsec in self.stdan_history_secs:
                scene = snapshots.get(np.round(hsec, 2), {})
                match = None
                for entry in scene.get("vehicles", []):
                    if entry["id"] == actor.id:
                        match = entry
                        break
                if match is None:
                    history = None
                    break
                centroid = match["centroid"]
                history.append([centroid[0], centroid[1], match["yaw"]])

            if history is not None:
                trackings[actor.id] = np.asarray(history, dtype=np.float64)
        return trackings

    def _make_stdan_predictions(self):
        selected_actor_idxs = self._selected_prediction_actor_idxs()
        if not selected_actor_idxs:
            return self._fake_target_prediction()

        trackings = self._stdan_trackings()
        tvs_positions = []
        tvs_mode_probs = []
        mus_per_tv = []
        sigmas_per_tv = []
        tvs_valid_pred = []
        prediction_debug = {
            "predictor": "stdan",
            "input_coordinate_system": "carla",
            "model_coordinate_system": "ngsim",
            "targets": [],
        }

        for actor_idx in selected_actor_idxs:
            actor = self.vehicle_actors[actor_idx]
            target_id = actor.id
            curr_position = self._actor_position_rhs(actor)
            tvs_positions.append(curr_position)

            enough_history = target_id in trackings and len(trackings[target_id]) >= self.stdan_predictor.full_length
            if enough_history:
                target_state = trackings[target_id][-1]
                try:
                    result = self.stdan_predictor.predict(
                        target_id,
                        target_state,
                        trackings,
                        num_modes=self.ego_num_modes,
                        horizon_steps=self.ego_N,
                    )
                    stdan_output = result["stdan_output"]
                    assert stdan_output.shape == (9, 50, 5)
                    tvs_mode_probs.append(result["mode_probabilities"])
                    mus_per_tv.append(result["mus"])
                    sigmas_per_tv.append(result["sigmas"])
                    tvs_valid_pred.append(True)
                    prediction_debug["targets"].append(
                        {
                            "actor_id": target_id,
                            "valid": True,
                            "stdan_output_shape": stdan_output.shape,
                            "stdan_output": stdan_output,
                            "probabilities": result["probabilities"],
                            "top_mode_indices": result["top_mode_indices"],
                            "mask_shape": tuple(result["mask"].shape),
                            "mask_true_count": int(result["mask"].sum().item()),
                        }
                    )
                    continue
                except Exception as exc:
                    prediction_debug["targets"].append(
                        {
                            "actor_id": target_id,
                            "valid": False,
                            "fallback_reason": repr(exc),
                        }
                    )
            else:
                prediction_debug["targets"].append(
                    {
                        "actor_id": target_id,
                        "valid": False,
                        "fallback_reason": "insufficient_history",
                    }
                )

            probs, mus, sigmas = self._constant_velocity_prediction(actor_idx, curr_position)
            tvs_mode_probs.append(probs)
            mus_per_tv.append(mus)
            sigmas_per_tv.append(sigmas)
            tvs_valid_pred.append(False)

        self._last_prediction_debug = prediction_debug
        tvs_mode_dists = [mus_per_tv, sigmas_per_tv]
        return tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred

    def _make_multipath_predictions(self):
        selected_actor_idxs = self._selected_prediction_actor_idxs()
        if not selected_actor_idxs:
            return self._fake_target_prediction()

        tvs_positions = []
        tvs_mode_probs = []
        mus_per_tv = []
        sigmas_per_tv = []
        tvs_valid_pred = []

        for actor_idx in selected_actor_idxs:
            target_agent_id = self.vehicle_actors[actor_idx].id
            past_states, R_target_to_world, t_target_to_world = \
                get_target_agent_history(self.agent_history, target_agent_id)

            curr_target_vehicle_position = R_target_to_world @ past_states[-1, 1:3] + t_target_to_world
            tvs_positions.append(curr_target_vehicle_position)

            if np.any(np.isnan(past_states)):
                probs, mus, sigmas = self._constant_velocity_prediction(actor_idx, curr_target_vehicle_position)
                tvs_mode_probs.append(probs)
                mus_per_tv.append(mus)
                sigmas_per_tv.append(sigmas)
                tvs_valid_pred.append(False)
            else:
                img_tv = self.rasterizer.rasterize(self.agent_history, target_agent_id)
                gmm_pred = self.pred_model.predict_instance(img_tv, past_states[:-1])
                gmm_pred.transform(R_target_to_world, t_target_to_world)
                gmm_pred = gmm_pred.get_top_k_GMM(self.ego_num_modes)

                tvs_mode_probs.append(gmm_pred.mode_probabilities)
                mus_per_tv.append(gmm_pred.mus[:, :self.ego_N, :])
                sigmas_per_tv.append(gmm_pred.sigmas[:, :self.ego_N, :, :])
                tvs_valid_pred.append(True)

        tvs_mode_dists = [mus_per_tv, sigmas_per_tv]
        return tvs_positions, tvs_mode_probs, tvs_mode_dists, tvs_valid_pred

    def _setup_carla_world(self, carla_params):
        client = carla.Client(carla_params.ip_addr, carla_params.port)
        client.set_timeout(carla_params.timeout_period)
        reuse_loaded_world = _env_flag("ACC_REUSE_CARLA_WORLD")
        world_key = (carla_params.ip_addr, int(carla_params.port), carla_params.map_str)
        if reuse_loaded_world:
            self.world = client.get_world()
            current_map = self.world.get_map().name
            if not str(current_map).endswith(carla_params.map_str):
                self.world = client.load_world(carla_params.map_str)
            _REUSED_CARLA_WORLD_KEYS.add(world_key)
        else:
            self.world = client.load_world(carla_params.map_str)
        self.world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

    def _setup_camera(self, drone_viz_params):
        bp_library = self.world.get_blueprint_library()
        bp_drone  = bp_library.find('sensor.camera.rgb')




        # This is like a top down view of the intersection.  Can tune later.
        # cam_loc = carla.Location(x=drone_viz_params.x,
        #                          y=drone_viz_params.y,
        #                          z=drone_viz_params.z)
        # cam_ori = carla.Rotation(roll=drone_viz_params.roll,
        #                  pitch=drone_viz_params.pitch,
        #                  yaw=drone_viz_params.yaw)
        ego_location = self.vehicle_actors[self.ego_vehicle_idx].get_location()
        self.drone_z_offset = drone_viz_params.z
        cam_loc = carla.Location(x=ego_location.x,
                                 y=ego_location.y,
                                 z=ego_location.z+self.drone_z_offset)
        cam_ori = carla.Rotation(yaw=drone_viz_params.yaw, pitch=drone_viz_params.pitch)

        self.drone_pitch=np.radians(float(drone_viz_params.pitch))
        self.drone_img_width=drone_viz_params.img_width
        self.drone_img_height=drone_viz_params.img_height
        self.drone_fov=np.radians(float(drone_viz_params.fov))

        cam_transform = carla.Transform(cam_loc, cam_ori)

        bp_drone.set_attribute('image_size_x', str(drone_viz_params.img_width))
        bp_drone.set_attribute('image_size_y', str(drone_viz_params.img_height))
        bp_drone.set_attribute('fov', str(drone_viz_params.fov))
        bp_drone.set_attribute('role_name', 'drone')

        self.drone = self.world.spawn_actor(bp_drone, cam_transform)

    def _setup_vehicles(self, vehicle_params_list, carla_params):
        intersection_fname = os.path.join( os.path.dirname(os.path.abspath(__file__)),
                                           carla_params.intersection_csv_loc )
        intersection = load_intersection(intersection_fname)
        bp_library = self.world.get_blueprint_library()

        self.vehicle_actors   = []
        self.vehicle_policies = []
        self.vehicle_colors   = []
        self.vehicle_init_speeds = []
        self.vehicle_actor_keys = []
        self.vehicle_active = []
        self.collision_sensors = []
        self.collision_events = []
        self.collision_sensor_errors = []
        self.retired_actor_records = []
        ego_vehicle_idxs  = []
        tv_vehicle_idxs   = []
        static_vehicle_idxs   = []

        for idx, vp in enumerate(vehicle_params_list):
            veh_bp = bp_library.find(vp.vehicle_type)
            veh_bp.set_attribute("color", vp.vehicle_color)
            veh_bp.set_attribute("role_name", vp.role)
            self.vehicle_colors.append([int(x) for x in vp.vehicle_color.split(", ")])

            if vp.role == "ego":
                ego_vehicle_idxs.append(idx)
            elif vp.role == "static":
                # pass
                static_vehicle_idxs.append(idx)
            elif "target" in vp.role:
                tv_vehicle_idxs.append(idx)
            elif "traffic" in vp.role:
                pass
            else:
                raise ValueError(f"Invalid vehicle role selection : {vp.role}")

            start_transform = get_intersection_transform(intersection, vp, "start")
            goal_transform  = get_intersection_transform(intersection, vp, "goal")

            veh_actor  = self.world.spawn_actor(veh_bp, start_transform)
            veh_actor.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))

            veh_policy = get_vehicle_policy(vp, veh_actor, goal_transform)

            self.vehicle_actors.append(veh_actor)
            self.vehicle_policies.append(veh_policy)
            self.vehicle_init_speeds.append(vp.init_speed)
            self.vehicle_actor_keys.append(f"{vp.role}_{idx}")
            self.vehicle_active.append(True)

        if len(ego_vehicle_idxs) != 1:
            raise RuntimeError(f"Invalid number of ego vehicles spawned: {len(ego_vehicle_idxs)}")
        self.ego_vehicle_idx = ego_vehicle_idxs[0]
        self.ego_N           = vehicle_params_list[self.ego_vehicle_idx].N
        self.ego_num_modes   = vehicle_params_list[self.ego_vehicle_idx].num_modes

        self.world.get_spectator().set_transform(carla.Transform(carla.Location(x=intersection[0][0][0]-10., y=intersection[0][0][1], z=5. ),
                                                                 carla.Rotation(pitch=-10., yaw=intersection[0][0][2])))

        # Note: this can be empty, as checked in the _make_predictions code.
        self.tv_vehicle_idxs = tv_vehicle_idxs
        self.static_vehicle_idxs = static_vehicle_idxs
        self._setup_collision_sensors()

    def _setup_collision_sensors(self):
        bp_library = self.world.get_blueprint_library()
        collision_bp = bp_library.find("sensor.other.collision")
        self.collision_sensors = []
        for idx, actor in enumerate(self.vehicle_actors):
            try:
                sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=actor)
                sensor.listen(lambda event, actor_idx=idx: self._record_collision_event(actor_idx, event))
            except RuntimeError as exc:
                sensor = None
                self.collision_sensor_errors.append({
                    "actor_index": int(idx),
                    "actor_key": self.vehicle_actor_keys[idx] if idx < len(self.vehicle_actor_keys) else None,
                    "error": repr(exc),
                })
            self.collision_sensors.append(sensor)

    def _record_collision_event(self, actor_idx, event):
        actor = self.vehicle_actors[actor_idx] if actor_idx < len(self.vehicle_actors) else None
        other_actor = getattr(event, "other_actor", None)
        if actor is not None and other_actor is not None and actor.id == other_actor.id:
            return
        impulse = getattr(event, "normal_impulse", None)
        impulse_x = float(getattr(impulse, "x", 0.0)) if impulse is not None else 0.0
        impulse_y = float(getattr(impulse, "y", 0.0)) if impulse is not None else 0.0
        impulse_z = float(getattr(impulse, "z", 0.0)) if impulse is not None else 0.0
        timestamp = getattr(event, "timestamp", None)
        time_s = float(timestamp) if isinstance(timestamp, (int, float)) else None
        if time_s is None:
            try:
                snapshot = self.world.get_snapshot()
                time_s = float(snapshot.timestamp.elapsed_seconds)
            except RuntimeError:
                time_s = None
        transform = getattr(event, "transform", None)
        location = getattr(transform, "location", None) if transform is not None else None
        self.collision_events.append({
            "frame": getattr(event, "frame", None),
            "time_s": time_s,
            "actor_index": int(actor_idx),
            "actor_key": self.vehicle_actor_keys[actor_idx] if actor_idx < len(self.vehicle_actor_keys) else None,
            "actor_id": int(actor.id) if actor is not None else None,
            "actor_role": actor.attributes.get("role_name") if actor is not None else None,
            "other_actor_id": int(other_actor.id) if other_actor is not None else None,
            "other_actor_type": getattr(other_actor, "type_id", None),
            "other_actor_role": (
                other_actor.attributes.get("role_name")
                if other_actor is not None and hasattr(other_actor, "attributes")
                else None
            ),
            "normal_impulse_x": impulse_x,
            "normal_impulse_y": impulse_y,
            "normal_impulse_z": impulse_z,
            "normal_impulse_norm": float(np.linalg.norm([impulse_x, impulse_y, impulse_z])),
            "x": float(location.x) if location is not None else None,
            "y": float(location.y) if location is not None else None,
            "z": float(location.z) if location is not None else None,
        })

    def _destroy_actor_quietly(self, actor):
        if actor is None:
            return
        try:
            if hasattr(actor, "is_alive") and not actor.is_alive:
                return
            actor.destroy()
        except RuntimeError:
            pass

    def _retire_done_actor(self, actor_idx, time_s, reason):
        if actor_idx == self.ego_vehicle_idx or not self.vehicle_active[actor_idx]:
            return
        actor = self.vehicle_actors[actor_idx]
        actor_key = self.vehicle_actor_keys[actor_idx]
        sensor = self.collision_sensors[actor_idx] if actor_idx < len(self.collision_sensors) else None
        if sensor is not None:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            self._destroy_actor_quietly(sensor)
            self.collision_sensors[actor_idx] = None
        try:
            actor.set_simulate_physics(False)
        except RuntimeError:
            pass
        if hasattr(actor, "set_enable_gravity"):
            try:
                actor.set_enable_gravity(False)
            except RuntimeError:
                pass
        if hasattr(actor, "set_actor_enable_collision"):
            try:
                actor.set_actor_enable_collision(False)
            except RuntimeError:
                pass
        self._destroy_actor_quietly(actor)
        self.vehicle_active[actor_idx] = False
        self.retired_actor_records.append({
            "time_s": float(time_s) if time_s is not None else None,
            "actor_index": int(actor_idx),
            "actor_key": actor_key,
            "actor_id": int(actor.id),
            "reason": reason,
        })

    def _setup_predictions(self, prediction_params):
        self.prediction_model_type = prediction_params.predictor_type.lower()
        self._last_prediction_debug = None

        if self.prediction_model_type == "stdan":
            from predictor.stdan.stdan_predictor import STDANPredictor

            ckpt_path = prediction_params.stdan_ckpt
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(rootdir, ckpt_path)
            self.stdan_predictor = STDANPredictor(
                ckpt_path=ckpt_path,
                history=prediction_params.stdan_history,
                future=prediction_params.stdan_future,
                dt=prediction_params.stdan_dt,
                input_coordinate_system=prediction_params.stdan_input_coordinates,
                mc_dropout=prediction_params.stdan_mc_dropout,
            )
            self.stdan_history_secs = [
                round(i * self.stdan_predictor.dt, 2)
                for i in range(self.stdan_predictor.full_length - 1, -1, -1)
            ]
            self.stdan_history_closeness_eps = self.stdan_predictor.dt / 2.0 + 1e-6
            history_max_length = max(
                20,
                int(np.ceil((max(self.stdan_history_secs) + self.stdan_predictor.dt) * self.carla_fps)) + 2,
            )
            self.agent_history = AgentHistory(
                self.world.get_actors(),
                history_max_length=history_max_length,
            )
            self.rasterizer = None
            self.pred_model = None
        elif self.prediction_model_type == "multipath":
            from models.deploy_multipath_model import DeployMultiPath

            self.agent_history = AgentHistory(self.world.get_actors())
            self.rasterizer    = SemBoxRasterizer(self.world.get_map().get_topology(), render_traffic_lights=\
                                                     prediction_params.render_traffic_lights)
            prefix             = os.path.abspath(__file__).split('carla')[0] + 'models/'
            self.pred_model    = DeployMultiPath(prefix+prediction_params.model_weights, \
                                                 np.load(prefix+prediction_params.model_anchors))

            # Try to do a sample prediction, initialize and check GPU model is working fine.
            blank_image = np.zeros((self.rasterizer.sem_rast.raster_height,
                                    self.rasterizer.sem_rast.raster_width,
                                    3), dtype=np.uint8)
            zero_traj   = np.column_stack(( np.arange(-1.0, 0.00, 0.2),
                                            np.zeros((5,3))
                                          )).astype(np.float32)
            self.pred_model.predict_instance(image_raw   = blank_image,
                                             past_states = zero_traj)
        else:
            raise ValueError(f"Unsupported predictor_type: {prediction_params.predictor_type}")

    @staticmethod
    def _split_mode_dists(tvs_mode_dists):
        if len(tvs_mode_dists) == 2 and len(tvs_mode_dists[1]) > 0:
            first_sigma_candidate = np.asarray(tvs_mode_dists[1][0])
            if first_sigma_candidate.ndim == 4:
                return tvs_mode_dists[0], tvs_mode_dists[1]
        return [entry[0] for entry in tvs_mode_dists], [entry[1] for entry in tvs_mode_dists]

    def _viz_gmm(self, img, tvs_mode_dists, mdist_sq_thresh=5.991):
        mus_per_tv, sigmas_per_tv = self._split_mode_dists(tvs_mode_dists)

        for k, (mus, sigmas) in enumerate(zip(mus_per_tv, sigmas_per_tv)):

            drone_location = self.drone.get_location()
            self.p_drone=np.array([drone_location.x,-drone_location.y, drone_location.z])
            tv_location = self.vehicle_actors[self.tv_vehicle_idxs[0]].get_location()
            self.tv_z=tv_location.z
            scale=self.drone_img_width/(2*self.p_drone[2]*np.tan(0.5*self.drone_fov))

            self.rot_world_to_drone=scale*np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])\
                                        @np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])\
                                        @np.array([[np.cos(-self.drone_pitch),0.,np.sin(-self.drone_pitch)],[0.,1.,0.],[-np.sin(-self.drone_pitch),0.,np.cos(-self.drone_pitch)]])



            # self.A_world_to_drone = np.array([[    0., -19.2],
            #                                   [-19.2,      0.]])
            # self.b_world_to_drone = -self.p_drone

            # Note: we reverse mode_dists and colors s.t. least probable mode is plotted first.
            # zip_obj = zip( reversed(mus), reversed(sigmas), reversed(self.mode_rgb_colors) )
            zip_obj = zip( mus, sigmas, self.mode_rgb_colors )

            for mean_traj, covar_traj, color in zip_obj:
                # color = color[::-1] # rgb to bgr
                for (mean_xy, covar_xy) in zip(mean_traj, covar_traj):

                    mu_px    = self.rot_world_to_drone @ (np.hstack((mean_xy,self.tv_z)) - self.p_drone)
                    center_x = int(mu_px[0]) + self.drone_img_width // 2
                    center_y = int(mu_px[1]) + self.drone_img_height // 2
                    covar_px = self.rot_world_to_drone[0:2,0:2] @ covar_xy @ self.rot_world_to_drone[0:2,0:2].T
                    evals_px, evecs_px = np.linalg.eigh(covar_px)
                    # import pdb;pdb.set_trace()
                    length_ax1 = int( np.sqrt(mdist_sq_thresh * evals_px[0]) ) # half the first axis diameter in pixels
                    length_ax2 = int( np.sqrt(mdist_sq_thresh * evals_px[1]) ) # half the second axis diameter in pixels
                    ang_1 = -np.degrees( np.arctan2(evecs_px[1,0], evecs_px[0,0]) ) # -ang since cv2.ellipse uses clockwise angle
                    cv2.ellipse( img, (center_x, center_y), (length_ax1, length_ax2), ang_1, 0, 360, color, thickness=2)

    def _viz_traj_hist(self, img, radius=2):
        for idx_act, (act, act_color) in enumerate(zip(self.vehicle_actors,self.vehicle_colors)):
            act_key = self.vehicle_actor_keys[idx_act]
            act_traj = self.results_dict[act_key]["state_trajectory"]
            act_color = act_color[::-1] # rgb to bgr

            for act_st in act_traj:
                act_xy = np.array(act_st[1:3])
                act_px = self.A_world_to_drone @ act_xy + self.b_world_to_drone
                center_x = int(act_px[0])
                center_y = int(act_px[1])
                cv2.circle(img, (center_x, center_y), radius, act_color, thickness=-1)
