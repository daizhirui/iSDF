# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import random
import sys
import cv2
import magnum as mn
import numpy as np
import json
import trimesh
from scipy.spatial.transform import Rotation as R
from datetime import datetime

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import (
    RedwoodDepthNoiseModel)


# Max value of uint16 is 65535.
max_depth = 20.
depth_factor = 65535 / max_depth


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    if "dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["dataset_config"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.allow_sliding = settings["allow_sliding"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.orientation = [
        settings["sensor_pitch"], 0.0, 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    # depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
    # depth_sensor_spec.noise_model_kwargs = dict(
    #     noise_multiplier=1.)
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"], 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"], settings["width"]]
        color_sensor_3rd_person_spec.position = [
            0.0, settings["sensor_height"] + 0.35, 0.5]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 6, 0, 0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement
    # in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.016)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# Visualisation functions ------------------------------------


# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, depth_obs=np.array([])):

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(
            np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))

    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, traj_points=None):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o",
                     markersize=10, alpha=0.8)
    if traj_points is not None:
        for point in traj_points:
            plt.plot(point[0], point[1], marker="o",
                     markersize=5, alpha=0.4, color="blue")
    return fig


# Save functions ----------------------------------------

def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(np.pi), -np.sin(np.pi), 0],
                  [0, np.sin(np.pi), np.cos(np.pi), 0],
                  [0, 0, 0, 1]])
    return transform @ T


# poses = {}


def save_now(agent, observations, i, save_img_path):
    state = agent.get_state()
    robot_pose = get_pose(state)
    sensor_state = state.sensor_states['color_sensor']
    sensor_pose = get_pose(sensor_state)

    rgb = observations["color_sensor"]
    depth = observations["depth_sensor"]
    # display_sample(rgb, depth)
    save_sample(rgb, depth, i, save_img_path)

    # poses[i] = {
    #     "robot": {
    #         "pos": state.position.tolist(),
    #         "quat": [state.rotation.real] + state.rotation.imag.tolist(),
    #     },
    #     "sensor": {
    #         "pos": sensor_state.position.tolist(),
    #         "quat": [sensor_state.rotation.real] + sensor_state.rotation.imag.tolist(),
    #     }
    # }

    i += 1

    return sensor_pose, robot_pose, i


def save_sample(img, depth, i, save_img_path):
    noiseModel = RedwoodDepthNoiseModel(noise_multiplier=2.5, gpu_device_id=0)
    noisy_depth = noiseModel(depth)
    noisy_depth[:, :80] = depth[:, :80]

    assert(depth.max() < max_depth)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    depth_save = (depth * depth_factor).astype(np.uint16)
    noisy_depth_save = (noisy_depth * depth_factor).astype(np.uint16)

    # vis = np.vstack((depth_save[::2, ::2], noisy_depth_save[::2, ::2]))
    # cv2.imshow("d", vis)
    # cv2.waitKey(0)

    cv2.imwrite(os.path.join(save_img_path, f"frame{i:06d}.png"), img)
    cv2.imwrite(os.path.join(save_img_path, f"depth{i:06d}.png"), depth_save)
    cv2.imwrite(os.path.join(save_img_path, f"ndepth{i:06d}.png"),
                noisy_depth_save)


def save_traj(sensor_poses, save_path):
    with open(os.path.join(save_path, "traj.txt"), 'w') as f:
        for p in sensor_poses:
            p = to_opengl_transform(p)
            for e in p.flatten():
                f.write(f"{e:.6f} ")
            f.write("\n")


def get_pose(state):
    pose = np.eye(4)
    pose[:3, 3] = state.position
    R = utils.quat_to_magnum(state.rotation).to_matrix()
    pose[:3, :3] = R

    return pose


class ContinuousPathFollower:
    def __init__(self, sim, path, agent_scene_node, waypoint_threshold):
        self._sim = sim
        self._points = path.points[:]
        assert len(self._points) > 0
        self._length = path.geodesic_distance
        self._node = agent_scene_node
        self._threshold = waypoint_threshold
        self._step_size = 0.01
        self.progress = 0  # geodesic distance -> [0,1]
        self.waypoint = path.points[0]

        # setup progress waypoints
        _point_progress = [0]
        _segment_tangents = []
        _length = self._length
        for ix, point in enumerate(self._points):
            if ix > 0:
                segment = point - self._points[ix - 1]
                segment_length = np.linalg.norm(segment)
                segment_tangent = segment / segment_length
                _point_progress.append(
                    segment_length / _length + _point_progress[ix - 1]
                )
                # t-1 -> t
                _segment_tangents.append(segment_tangent)
        self._point_progress = _point_progress
        self._segment_tangents = _segment_tangents
        # final tangent is duplicated
        self._segment_tangents.append(self._segment_tangents[-1])

        print("self._length = " + str(self._length))
        print("num points = " + str(len(self._points)))
        print("self._point_progress = " + str(self._point_progress))
        print("self._segment_tangents = " + str(self._segment_tangents))

    def pos_at(self, progress):
        if progress <= 0:
            return self._points[0]
        elif progress >= 1.0:
            return self._points[-1]

        path_ix = 0
        for ix, prog in enumerate(self._point_progress):
            if prog > progress:
                path_ix = ix
                break

        segment_distance = self._length * (progress
            - self._point_progress[path_ix - 1])
        return (
            self._points[path_ix - 1]
            + self._segment_tangents[path_ix - 1] * segment_distance
        )

    def update_waypoint(self):
        # Waypoint is pushed away from agent as it moves along the path
        if self.progress < 1.0:
            wp_disp = self.waypoint - self._node.absolute_translation
            wp_dist = np.linalg.norm(wp_disp)
            node_pos = self._node.absolute_translation
            step_size = self._step_size
            threshold = self._threshold
            while wp_dist < threshold:
                self.progress += step_size
                self.waypoint = self.pos_at(self.progress)
                if self.progress >= 1.0:
                    break
                wp_disp = self.waypoint - node_pos
                wp_dist = np.linalg.norm(wp_disp)


def get_navigable_pt(topdown_map, sim, meters_per_pixel):
    found = False
    while not found:
        pt = sim.pathfinder.get_random_navigable_point()[None, :]
        xy = convert_points_to_topdown(
            sim.pathfinder, pt, meters_per_pixel)
        xy = np.round(xy).astype(int)
        found = topdown_map[xy[0, 1], xy[0, 0]]
    return pt[0]


def track_waypoint(waypoint, rs, vc, dt=1.0 / 60.0):
    # Sets velocity control structure velocity.
    # Is then integrated by agent.
    angular_error_threshold = 0.5
    max_linear_speed = 0.45
    max_turn_speed = 0.4
    glob_forward = rs.rotation.transform_vector(
        mn.Vector3(0, 0, -1.0)).normalized()
    glob_right = rs.rotation.transform_vector(
        mn.Vector3(-1.0, 0, 0)).normalized()
    to_waypoint = mn.Vector3(waypoint) - rs.translation
    u_to_waypoint = to_waypoint.normalized()
    angle_error = float(mn.math.angle(glob_forward, u_to_waypoint))

    new_velocity = 0
    if angle_error < angular_error_threshold:
        # speed up to max
        new_velocity = (vc.linear_velocity[2] - max_linear_speed) / 2.0
    else:
        # slow down to 0
        new_velocity = (vc.linear_velocity[2]) / 2.0
    vc.linear_velocity = mn.Vector3(0, 0, new_velocity)

    # angular part
    rot_dir = 1.0
    if mn.math.dot(glob_right, u_to_waypoint) < 0:
        rot_dir = -1.0
    angular_correction = 0.0
    if angle_error > (max_turn_speed * 10.0 * dt):
        angular_correction = max_turn_speed
    else:
        angular_correction = angle_error / 2.0

    vc.angular_velocity = mn.Vector3(
        0,
        np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed),
        0
    )


def floodfill(mat, start):
    """ Converts all components connected to the start point
        to zeros.
    """
    print("Doing floodfill algorithm by BFS")
    extent = mat.shape
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    queue = [start]
    queued = mat.copy()

    while queue:
        ix = queue.pop(0)
        mat[ix[1], ix[0]] = 0
        queued[ix[1], ix[0]] = 0

        neighbours = ix + directions
        for ne in neighbours:
            if ne[0] >= 0 and ne[0] < extent[1]:
                if ne[0] >= 0 and ne[1] < extent[0]:
                    if mat[ne[1], ne[0]] == 1 and queued[ne[1], ne[0]] == 1:
                        queue.append(ne)
                        queued[ne[1], ne[0]] = 0

    return mat


def gen_navigation_seq(
    scene,
    seed,
    n_waypoints,
    save_path,
    replica_cad_path,
    habitat_sim_path,
    seq_name=None,
    save_video=False,
    agent_radius=0.35,
):

    meters_per_pixel = 0.02
    dataset_config = replica_cad_path + "replicaCAD.scene_dataset_config.json"
    scene = replica_cad_path + scene

    if seq_name is None:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        seq_name = f"nav_{time_str}/"
    save_path = save_path + seq_name + "/"
    save_img_path = save_path + "/results/"
    os.makedirs(save_path, exist_ok=False)
    os.makedirs(save_img_path, exist_ok=False)

    sim_settings = {
        "dataset_config": dataset_config,
        "scene": scene,  # Sscene path
        "enable_physics": True,  # Need for objects
        "allow_sliding": True,

        "default_agent": 0,  # Index of the default agent
        "sensor_height": 0.6,  # Height of sensors in meters
        "sensor_pitch": 0.0,  # sensor pitch (x rotation in rads)
        "color_sensor": True,  # RGB sensor
        "depth_sensor": True,  # Depth sensor
        "width": 1200,  # Spatial resolution of the observations
        "height": 680,
        "color_sensor_3rd_person": True,  # RGB sensor 3rd person
    }

    # Create sim

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])

    random.seed(seed)
    sim.seed(seed)
    np.random.seed(seed)

    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(habitat_sim_path, "data/objects/locobot_merged")))
    rigid_obj_mgr = sim.get_rigid_object_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    art_obj_mgr = sim.get_articulated_object_manager()

    # Set all objects to static so they are non-navigable in navmesh
    rigid_objs = rigid_obj_mgr.get_objects_by_handle_substring()
    for k in rigid_objs.keys():
        rigid_objs[k].motion_type = habitat_sim.physics.MotionType.STATIC
    art_objs = art_obj_mgr.get_objects_by_handle_substring()
    for k in art_objs.keys():
        art_objs[k].motion_type = habitat_sim.physics.MotionType.STATIC

    # load the locobot_merged asset
    locobot_template_handle = obj_attr_mgr.get_file_template_handles("locobot")[0]
    # add robot object to the scene with the agent/camera SceneNode attached
    locobot_obj = rigid_obj_mgr.add_object_by_template_handle(
        locobot_template_handle, sim.agents[0].scene_node)
    # set the agent's body to kinematic since we will be updating position manually
    locobot_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    # create and configure a new VelocityControl structure
    # Note: this is NOT the object's VelocityControl,
    # so it will not be consumed automatically in sim.step_physics
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True

    default_nav_mesh_settings = habitat_sim.NavMeshSettings()
    default_nav_mesh_settings.set_defaults()
    inflated_nav_mesh_settings = habitat_sim.NavMeshSettings()
    inflated_nav_mesh_settings.set_defaults()
    inflated_nav_mesh_settings.agent_radius = agent_radius
    inflated_nav_mesh_settings.agent_height = sim_settings["sensor_height"]
    recompute_successful = sim.recompute_navmesh(
        sim.pathfinder, inflated_nav_mesh_settings, include_static_objects=True)
    if not recompute_successful:
        print("Failed to recompute navmesh!")

    # Get navmesh
    floor_height = sim.pathfinder.get_bounds()[0][1]  # min y value (i.e. floor)
    topdown_map_floor = sim.pathfinder.get_topdown_view(
        meters_per_pixel, floor_height)
    topdown_map_sensor = sim.pathfinder.get_topdown_view(
        meters_per_pixel, sim_settings["sensor_height"])
    # display_map(topdown_map_floor)
    # display_map(topdown_map_sensor)

    topdown_map = np.logical_and(
        topdown_map_floor, topdown_map_sensor).astype(np.uint8)
    # topdown_map = maps.get_topdown_map(
    #     sim.pathfinder, height, meters_per_pixel=meters_per_pixel)

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    topdown_map_col = recolor_map[topdown_map]

    # get shortest path to the object from the agent position
    paths = []
    start = get_navigable_pt(topdown_map.astype(bool), sim, meters_per_pixel)
    waypoints = [start]

    for i in range(n_waypoints):
        found_path = False
        while not found_path:
            # e.g. to avoid waypoint on a navigable island
            end = get_navigable_pt(
                topdown_map.astype(bool), sim, meters_per_pixel)
            path = habitat_sim.ShortestPath()
            path.requested_start = waypoints[-1]
            path.requested_end = end
            found_path = sim.pathfinder.find_path(path)
            if found_path:
                waypoints.append(path.requested_end)
                paths.append(path)

    # Vis generated waypoints
    waypoints_xy = convert_points_to_topdown(
        sim.pathfinder, waypoints, meters_per_pixel)
    fig1 = display_map(topdown_map_col, key_points=waypoints_xy)
    plt.show()

    locobot_obj.translation = start

    # Recompute uninflated navmesh for executing navigation
    recompute_successful = sim.recompute_navmesh(
        sim.pathfinder, default_nav_mesh_settings, include_static_objects=True)
    if not recompute_successful:
        print("Failed to recompute navmesh 2!")

    topdown_map = sim.pathfinder.get_topdown_view(
        meters_per_pixel, floor_height).astype(np.uint8)
    start_xy = convert_points_to_topdown(
        sim.pathfinder, [waypoints[0]], meters_per_pixel)
    islands = floodfill(topdown_map, start_xy[0].astype(int))

    display_map(recolor_map[islands])
    # plt.show()

    np.savetxt(save_path + "/unnavigable.txt", islands)
    bounds = sim.pathfinder.get_bounds()
    min_xy = np.array([bounds[0][0], bounds[0][2], meters_per_pixel])
    np.savetxt(os.path.join(save_path, "bounds.txt"), min_xy)

    print("save path", save_path)
    print("\n\n\ndepth_factor", depth_factor)

    time_step = 1.0 / 30.0
    save_ix = 0
    sensor_poses = []
    robot_poses = []
    observations = []

    # 2 seconds
    start_time = sim.get_world_time()
    while sim.get_world_time() - start_time < 2.0:
        sim.step_physics(time_step)
        obs = sim.get_sensor_observations()
        observations.append(obs)
        pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
        sensor_poses.append(pose)

    for path in paths:
        continuous_path_follower = ContinuousPathFollower(
            sim, path, locobot_obj.root_scene_node, waypoint_threshold=0.4)

        # manually control the object's kinematic state via
        # velocity integration
        start_time = sim.get_world_time()
        max_time = 30.0
        while (
            continuous_path_follower.progress < 1.0
            and sim.get_world_time() - start_time < max_time
        ):
            continuous_path_follower.update_waypoint()
            previous_rigid_state = locobot_obj.rigid_state

            # set velocities based on relative waypoint position/direction
            track_waypoint(
                continuous_path_follower.waypoint, previous_rigid_state,
                vel_control, dt=time_step)

            # manually integrate the rigid state
            target_rigid_state = vel_control.integrate_transform(
                time_step, previous_rigid_state)

            # snap rigid state to navmesh and set state to object/agent
            end_pos = sim.step_filter(
                previous_rigid_state.translation,
                target_rigid_state.translation
            )
            locobot_obj.translation = end_pos
            locobot_obj.rotation = target_rigid_state.rotation

            # Check if a collision occured
            dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation
            ).dot()
            dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation
            ).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0
            # when a collision _didn't_ happen. One case is going up stairs.
            # Instead, we check to see if the amount moved after application
            # of the filter is _less_ than the amount moved before the
            # application of the filter
            EPS = 1e-5
            collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

            # run any dynamics simulation
            sim.step_physics(time_step)

            # save observation
            obs = sim.get_sensor_observations()
            observations.append(obs)
            pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
            sensor_poses.append(pose)

    save_traj(sensor_poses, save_path)
    sensor_positions = [t[:3, 3] for t in sensor_poses]

    # convert world trajectory points to maps module grid points
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in sensor_positions
    ]
    maps.draw_path(topdown_map_col, trajectory)

    traj_points = convert_points_to_topdown(
        sim.pathfinder, sensor_positions, meters_per_pixel)

    fig = display_map(
        topdown_map_col, key_points=waypoints_xy, traj_points=traj_points)
    plt.savefig(save_path + "topdown.png")

    if save_video:
        # video rendering with embedded 1st person view
        overlay_dims = (
            int(sim_settings["width"] / 5),
            int(sim_settings["height"] / 5)
        )
        print("overlay_dims = " + str(overlay_dims))
        overlay_settings = [
            {
                "obs": "color_sensor",
                "type": "color",
                "dims": overlay_dims,
                "pos": (10, 10),
                "border": 2,
            },
            {
                "obs": "depth_sensor",
                "type": "depth",
                "dims": overlay_dims,
                "pos": (10, 30 + overlay_dims[1]),
                "border": 2,
            },
        ]
        print("overlay_settings = " + str(overlay_settings))

        vut.make_video(
            observations=observations,
            primary_obs="color_sensor_3rd_person",
            primary_obs_type="color",
            video_file=save_path + "vid",
            fps=int(1.0 / time_step),
            open_vid=False,
            overlay_settings=overlay_settings,
            depth_clip=10.0,
        )


def gen_manipulation_seq(
    scene,
    target_objs,
    occurence_ixs,
    seed,
    save_path,
    replica_cad_path,
    habitat_sim_path,
    seq_name=None,
    joints_config=None,
    sensor_height=1.3,
    approach_object=False,
    obj_offset=np.zeros(3),
    approach_t=0.75,
    save_bounds=True,
    save_video=False,
):
    target_obj_handles = [
        f"{x}_:{i:04d}" for (x, i) in zip(target_objs, occurence_ixs)
    ]

    meters_per_pixel = 0.02
    dataset_config = replica_cad_path + "replicaCAD.scene_dataset_config.json"
    scene = replica_cad_path + scene

    if seq_name is None:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        seq_name = f"nav_{time_str}"
    save_path = save_path + seq_name + "/"
    save_img_path = save_path + "/results/"
    os.makedirs(save_path, exist_ok=False)
    os.makedirs(save_img_path, exist_ok=False)

    if save_bounds:
        bounds_outfile = os.path.join(save_path, "obj_bounds.txt")
        save_obj_bounds(
            bounds_outfile, scene, replica_cad_path,
            target_objs, occurence_ixs=occurence_ixs)

    sim_settings = {
        "dataset_config": dataset_config,
        "scene": scene,  # Sscene path
        "enable_physics": True,  # Need for objects
        "allow_sliding": True,

        "default_agent": 0,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters
        "sensor_pitch": 0.0,  # sensor pitch (x rotation in rads)
        "color_sensor": True,  # RGB sensor
        "depth_sensor": True,  # Depth sensor
        "width": 1200,  # Spatial resolution of the observations
        "height": 680,
        "color_sensor_3rd_person": True,  # RGB sensor 3rd person
    }

    # Create sim

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])

    random.seed(seed)
    sim.seed(seed)
    np.random.seed(seed)

    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(
        str(os.path.join(habitat_sim_path, "data/objects/locobot_merged")))
    rigid_obj_mgr = sim.get_rigid_object_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    art_obj_mgr = sim.get_articulated_object_manager()

    if joints_config is not None:
        for k in joints_config.keys():
            art_obj_mgr.get_object_by_handle(
                k).joint_positions = joints_config[k]

    # Set all objects to static so they are non-navigable in navmesh
    rigid_objs = rigid_obj_mgr.get_objects_by_handle_substring()
    for k in rigid_objs.keys():
        rigid_objs[k].motion_type = habitat_sim.physics.MotionType.STATIC
    art_objs = art_obj_mgr.get_objects_by_handle_substring()
    for k in art_objs.keys():
        art_objs[k].motion_type = habitat_sim.physics.MotionType.STATIC

    # load the locobot_merged asset
    locobot_template_handle = obj_attr_mgr.get_file_template_handles("locobot")[0]
    # add robot object to the scene with the agent/camera SceneNode attached
    locobot_obj = rigid_obj_mgr.add_object_by_template_handle(
        locobot_template_handle, sim.agents[0].scene_node)
    # set the agent's body to kinematic since we will be updating position manually
    locobot_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    # create and configure a new VelocityControl structure
    # Note: this is NOT the object's VelocityControl,
    # so it will not be consumed automatically in sim.step_physics
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True

    default_nav_mesh_settings = habitat_sim.NavMeshSettings()
    default_nav_mesh_settings.set_defaults()
    inflated_nav_mesh_settings = habitat_sim.NavMeshSettings()
    inflated_nav_mesh_settings.set_defaults()
    inflated_nav_mesh_settings.agent_radius = 0.25
    inflated_nav_mesh_settings.agent_height = sim_settings["sensor_height"]
    recompute_successful = sim.recompute_navmesh(
        sim.pathfinder, inflated_nav_mesh_settings, include_static_objects=True)
    if not recompute_successful:
        print("Failed to recompute navmesh!")

    # Get navmesh
    floor_height = sim.pathfinder.get_bounds()[0][1]  # min y value (i.e. floor)
    topdown_map = sim.pathfinder.get_topdown_view(
        meters_per_pixel, floor_height).astype(np.uint8)

    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
    topdown_map_col = recolor_map[topdown_map]

    # get shortest path to the object from the agent position
    target_objs = [
        rigid_obj_mgr.get_object_by_handle(h)
        for h in target_obj_handles
    ]
    target_loc = target_objs[0].translation

    # choose start to be furthest from first object
    starts = [
        get_navigable_pt(topdown_map.astype(bool), sim, meters_per_pixel)
        for i in range(12)
    ]
    starts = [x for x in starts if x[1] < 0.6]
    diffs = [(target_loc - x).length() for x in starts]
    start = starts[np.argmax(diffs)]

    paths = []
    waypoints = [start]
    for obj in target_objs:
        path = habitat_sim.ShortestPath()
        path.requested_start = waypoints[-1]
        path.requested_end = obj.translation
        found_path = sim.pathfinder.find_path(path)

        if found_path:
            paths.append(path)
            waypoints.append(obj.translation)
        else:
            raise ValueError('Failed to find path to object.')

    # Vis generated waypoints
    waypoints_xy = convert_points_to_topdown(
        sim.pathfinder, waypoints, meters_per_pixel)
    fig1 = display_map(topdown_map_col, key_points=waypoints_xy)
    # plt.show()

    # set agent state to look at first object
    locobot_obj.translation = start
    vis_target = np.array([target_loc.x, target_loc.y, target_loc.z])
    vis_target[1] = start[1]
    locobot_obj.rotation = mn.Quaternion.from_matrix(
        mn.Matrix4.look_at(
            start, vis_target, np.array([0, 1.0, 0])  # up
        ).rotation()
    )

    # Recompute uninflated navmesh for executing navigation
    recompute_successful = sim.recompute_navmesh(
        sim.pathfinder, default_nav_mesh_settings, include_static_objects=True)
    if not recompute_successful:
        print("Failed to recompute navmesh 2!")

    print("\n\n\ndepth_factor", depth_factor)

    time_step = 1.0 / 30.0
    save_ix = 0
    sensor_poses = []
    observations = []

    # 2 seconds
    start_time = sim.get_world_time()
    while sim.get_world_time() - start_time < 2.0:
        sim.step_physics(time_step)
        obs = sim.get_sensor_observations()
        observations.append(obs)
        pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
        sensor_poses.append(pose)

    for i, path in enumerate(paths):
        target_loc = path.requested_end

        continuous_path_follower = ContinuousPathFollower(
            sim, path, locobot_obj.root_scene_node, waypoint_threshold=0.4)

        # manually control the object's kinematic state via
        # velocity integration
        start_time = sim.get_world_time()
        max_time = 30.0
        while (
            continuous_path_follower.progress < 1.0
            and sim.get_world_time() - start_time < max_time
        ):
            continuous_path_follower.update_waypoint()
            previous_rigid_state = locobot_obj.rigid_state

            # set velocities based on relative waypoint position/direction
            track_waypoint(
                continuous_path_follower.waypoint, previous_rigid_state,
                vel_control, dt=time_step)

            # manually integrate the rigid state
            target_rigid_state = vel_control.integrate_transform(
                time_step, previous_rigid_state)

            # snap rigid state to navmesh and set state to object/agent
            end_pos = sim.step_filter(
                previous_rigid_state.translation,
                target_rigid_state.translation
            )
            locobot_obj.translation = end_pos
            locobot_obj.rotation = target_rigid_state.rotation

            # Check if a collision occured
            dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation
            ).dot()
            dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation
            ).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0
            # when a collision _didn't_ happen. One case is going up stairs.
            # Instead, we check to see if the amount moved after application
            # of the filter is _less_ than the amount moved before the
            # application of the filter
            EPS = 1e-5
            collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

            # run any dynamics simulation
            sim.step_physics(time_step)

            # save observation
            obs = sim.get_sensor_observations()
            observations.append(obs)
            pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
            sensor_poses.append(pose)

        # Rotate sensor to look at target object
        vis_target = target_loc.copy()
        sensor_pos = agent.get_state().sensor_states['color_sensor'].position
        target_rot = mn.Quaternion.from_matrix(
            mn.Matrix4.look_at(
                sensor_pos, vis_target, np.array([0, 1.0, 0])
            ).rotation()
        )
        state = agent.get_state()
        sensor_rot = utils.quat_to_magnum(
            state.sensor_states['color_sensor'].rotation)
        for t in np.linspace(0, 1, 30):
            new_rot = mn.math.slerp_shortest_path(sensor_rot, target_rot, t)
            new_rot_q = utils.quat_from_magnum(new_rot)

            state = agent.get_state()
            state.sensor_states['color_sensor'].rotation = new_rot_q
            state.sensor_states['depth_sensor'].rotation = new_rot_q
            agent.set_state(state, infer_sensor_states=False)

            obs = sim.get_sensor_observations()
            observations.append(obs)
            pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
            sensor_poses.append(pose)

        # 2 seconds after arrival
        start_time = sim.get_world_time()
        while sim.get_world_time() - start_time < 2.0:
            sim.step_physics(time_step)
            obs = sim.get_sensor_observations()
            observations.append(obs)
            pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
            sensor_poses.append(pose)

        if approach_object:
            target_pos = target_objs[0].translation
            nav_pos = target_objs[0].translation + obj_offset
            sens_pos = agent.get_state().sensor_states['color_sensor'].position
            for t in np.linspace(0, approach_t, 50):
                new_pos = sens_pos + t * (nav_pos - sens_pos)
                new_rot = mn.Quaternion.from_matrix(mn.Matrix4.look_at(
                    new_pos, target_pos, np.array([0, 1.0, 0])).rotation())
                new_rot_q = utils.quat_from_magnum(new_rot)
                state = agent.get_state()
                state.sensor_states['color_sensor'].position = new_pos
                state.sensor_states['depth_sensor'].position = new_pos
                state.sensor_states['color_sensor'].rotation = new_rot_q
                state.sensor_states['depth_sensor'].rotation = new_rot_q
                agent.set_state(state, infer_sensor_states=False)

                obs = sim.get_sensor_observations()
                observations.append(obs)
                pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
                sensor_poses.append(pose)

            # 2 seconds after arrival
            start_time = sim.get_world_time()
            while sim.get_world_time() - start_time < 2.0:
                sim.step_physics(time_step)
                obs = sim.get_sensor_observations()
                observations.append(obs)
                pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
                sensor_poses.append(pose)

        if i < len(paths) - 1:
            # Rotate sensor to back to default orientation
            for t in np.linspace(0, 1, 30):
                ret_rot = mn.math.slerp_shortest_path(new_rot, sensor_rot, t)
                ret_rot_q = utils.quat_from_magnum(ret_rot)

                state = agent.get_state()
                state.sensor_states['color_sensor'].rotation = ret_rot_q
                state.sensor_states['depth_sensor'].rotation = ret_rot_q
                agent.set_state(state, infer_sensor_states=False)

                obs = sim.get_sensor_observations()
                observations.append(obs)
                pose, robot_pose, save_ix = save_now(agent, obs, save_ix, save_img_path)
                sensor_poses.append(pose)

    save_traj(sensor_poses, save_path)
    sensor_positions = [t[:3, 3] for t in sensor_poses]

    # convert world trajectory points to maps module grid points
    grid_dimensions = (topdown_map.shape[0], topdown_map.shape[1])
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in sensor_positions
    ]
    maps.draw_path(topdown_map_col, trajectory)

    traj_points = convert_points_to_topdown(
        sim.pathfinder, sensor_positions, meters_per_pixel)

    fig = display_map(
        topdown_map_col, key_points=waypoints_xy, traj_points=traj_points)
    plt.savefig(save_path + "topdown.png")

    if save_video:
        # video rendering with embedded 1st person view
        overlay_dims = (
            int(sim_settings["width"] / 5),
            int(sim_settings["height"] / 5)
        )
        print("overlay_dims = " + str(overlay_dims))
        overlay_settings = [
            {
                "obs": "color_sensor",
                "type": "color",
                "dims": overlay_dims,
                "pos": (10, 10),
                "border": 2,
            },
            {
                "obs": "depth_sensor",
                "type": "depth",
                "dims": overlay_dims,
                "pos": (10, 30 + overlay_dims[1]),
                "border": 2,
            },
        ]
        print("overlay_settings = " + str(overlay_settings))

        vut.make_video(
            observations=observations,
            primary_obs="color_sensor_3rd_person",
            primary_obs_type="color",
            video_file=save_path + "vid",
            fps=int(1.0 / time_step),
            open_vid=False,
            overlay_settings=overlay_settings,
            depth_clip=10.0,
        )


def get_transf_and_scale(conf):
    transform = np.eye(4)
    if "translation" in conf.keys():
        transform[:3, 3] = conf["translation"]

    if "rotation" in conf.keys():
        # Rotation is stored with different quaterion convention to scipy
        q = np.roll(conf["rotation"], -1)
        r = R.from_quat(q)
        transform[:3, :3] = r.as_matrix()

    scale = 1.
    if "uniform_scale" in conf.keys():
        scale = conf["uniform_scale"]

    return transform, scale


def load_mesh(conf, dataset_path, dump=True):
    fname = os.path.join(dataset_path, conf["template_name"] + ".glb")
    mesh = trimesh.load(fname)

    if isinstance(mesh, trimesh.Scene) and dump:
        mesh = mesh.dump().sum()

    transform, scale = get_transf_and_scale(conf)

    mesh.apply_scale(scale)
    mesh.apply_transform(transform)

    return mesh


def save_obj_bounds(bounds_outfile, scene_config, dataset_path, target_objs,
                    occurence_ixs=None, min_size=1e-5):
    if occurence_ixs is None:
        occurence_ixs = np.zeros(len(target_objs), dtype=int)

    with open(scene_config, 'r') as f:
        conf = json.load(f)
    obj_confs = conf["object_instances"]

    obj_bounds = []
    for i in range(len(target_objs)):
        obj = [
            x for x in obj_confs if
            x['template_name'] == "objects/" + target_objs[i]
        ][occurence_ixs[i]]
        mesh = load_mesh(obj, dataset_path, dump=True)
        bounds = mesh.bounds

        if np.max(mesh.bounds[1] - mesh.bounds[0]) < min_size:
            center = (bounds[1] + bounds[0]) / 2.
            offset = bounds - center
            offset = offset * 0.3 / np.max(mesh.bounds[1] - mesh.bounds[0])
            bounds = offset + center

        # trimesh.Scene(
        #     [mesh, mesh.bounding_box, trimesh.PointCloud(bounds)]).show()

        obj_bounds.append(bounds)

    with open(bounds_outfile, "w") as f:
        for bounds in obj_bounds:
            for e in bounds.flatten():
                f.write(str(e) + " ")
            f.write("\n")


if __name__ == "__main__":

    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--save_path', type=str, required=True)
    # parser.add_argument('--replica_cad_path', type=str, required=True)
    # parser.add_argument('--habitat_sim_path', type=str, required=True)
    # parser.add_argument('--seq_name', type=str, required=True)

    # args = parser.parse_args()

    # seq_name = args.seq_name
    # save_path = args.save_path
    # replica_cad_path = args.replica_cad_path
    # habitat_sim_path = args.habitat_sim_path

    seq_name = "apt_2_nav"
    save_path = "/home/joe/projects/incSDF/incSDF/data/seqs/"
    replica_cad_path = "/mnt/sda/ReplicaCAD/replica_cad/"
    habitat_sim_path = "/home/joe/projects/incSDF/habitat-sim/"

    # apt 2 navigation sequence

    if seq_name == "apt_2_nav":
        scene = "configs/scenes/apt_2.scene_instance.json"
        seed = 6
        n_waypoints = 4

        gen_navigation_seq(
            scene,
            seed,
            n_waypoints,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            # seq_name=seq_name,
            save_video=True,
        )

    # apt 3 navigation sequence

    if seq_name == "apt_3_nav":
        scene = "configs/scenes/apt_3.scene_instance.json"
        seed = 8
        n_waypoints = 8

        gen_navigation_seq(
            scene,
            seed,
            n_waypoints,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            # seq_name=seq_name,
            save_video=True,
            agent_radius=0.45,
        )

    # apt 2 object sequence

    if seq_name == "apt_2_obj":
        scene = "configs/scenes/apt_2.scene_instance.json"
        target_objs = [
            "frl_apartment_cup_02",
            "frl_apartment_lamp_02"
        ]
        occurence_ixs = [0, 0]
        seed = 50

        gen_manipulation_seq(
            scene,
            target_objs,
            occurence_ixs,
            seed,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            seq_name=seq_name,
            save_video=True,
            save_bounds=True,
        )

    # apt 3 object sequence

    if seq_name == "apt_3_obj":
        scene = "configs/scenes/apt_3.scene_instance.json"
        target_objs = [
            "frl_apartment_handbag",
            "frl_apartment_lamp_02"
        ]
        occurence_ixs = [0, 0]
        seed = 20

        gen_manipulation_seq(
            scene,
            target_objs,
            occurence_ixs,
            seed,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            # seq_name=seq_name,
            save_video=True,
            save_bounds=True,
        )

    # apt 2 manipulation sequence

    if seq_name == "apt_2_mnp":
        scene = "configs/scenes/apt_2_v1.scene_instance.json"
        target_objs = [
            "frl_apartment_kitchen_utensil_03",
        ]
        occurence_ixs = [0]
        joints_config = {"fridge_:0000": [0., np.pi / 2.]}
        seed = 3  # 15
        sensor_height = 1.12

        gen_manipulation_seq(
            scene,
            target_objs,
            occurence_ixs,
            seed,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            seq_name=seq_name,
            save_video=True,
            save_bounds=True,
            joints_config=joints_config,
            sensor_height=sensor_height,
            approach_object=True,
            obj_offset=np.array([0., 0., 0.2]),
            approach_t=0.90,
        )

    # apt 3 manipulation sequence

    if seq_name == "apt_3_mnp":
        scene = "configs/scenes/apt_3_v1.scene_instance.json"
        target_objs = [
            "frl_apartment_bowl_03",
        ]
        occurence_ixs = [0]
        joints_config = {
            "kitchen_counter_:0000": [0., 0., 0., 0., 0.38, 0., 0.]
        }
        seed = 10
        sensor_height = 1.12

        gen_manipulation_seq(
            scene,
            target_objs,
            occurence_ixs,
            seed,
            save_path,
            replica_cad_path,
            habitat_sim_path,
            seq_name=seq_name,
            save_video=True,
            save_bounds=True,
            joints_config=joints_config,
            sensor_height=sensor_height,
            approach_object=True,
            obj_offset=np.array([0., 0., -0.2]),
            approach_t=0.76,
        )
