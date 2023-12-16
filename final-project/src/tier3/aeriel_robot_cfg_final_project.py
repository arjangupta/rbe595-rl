# Modified by: Arjan Gupta & Taylor Bergeron

# Original copyright:
# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the aeriel_gym_simulator project.

from aerial_gym.envs.base.base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
WALL_SEMANTIC_ID = 8

class AerialRobotCfgFinalProjectTier3(BaseConfig):

    seed = 1

    class env:
        num_envs = 64
        num_observations = 13
        get_privileged_obs = True  # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4
        env_spacing = 1
        episode_length_s = 1.21e+6 # episode length in seconds (14 days)
        num_control_steps_per_env_step = 1 # number of physics steps per env step
        enable_onboard_cameras = True  # enable onboard cameras
        reset_on_collision = True  # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = True  # create a ground plane

        """For the walls, if at 0,0,0, will appear with half above ground plane, half below. The 
        wall is 20x20. The location of the wall is from its center, so to have the bottom
        flush with the ground, it needs to be 10z"""

    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt = 0.01
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 1  # 0: never, 1: last sub-step, 2: all sub-steps (default=2) # FIXME: this is 0 in tier1

    class control:
        """
        Control parameters
        controller:
            lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
            lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
            lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
        kP: gains for position
        kV: gains for velocity
        kR: gains for attitude
        kOmega: gains for angular velocity
        """
        controller = "lee_position_control"  # or "lee_velocity_control" or "lee_attitude_control"
        kP = [0.8, 0.8, 1.0]  # used for lee_position_control only
        kV = [0.5, 0.5, 0.4]  # used for lee_position_control, lee_velocity_control only
        kR = [3.0, 3.0, 1.0]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
        kOmega = [0.5, 0.5, 1.20]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
        scale_input = [1.0, 1.0, 1.0, 1.0]  # scale the input to the controller from -1 to 1 for each dimension, scale from -np.pi to np.pi for yaw in the case of position control

    class robot_asset:
        file = "{AERIAL_GYM_ROOT_DIR}/resources/robots/quad/model.urdf"
        name = "aerial_robot"  # actor name
        base_link_name = "base_link"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter #FIXME: in tier1 this is 1
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001

    class asset_state_params(robot_asset):
        num_assets = 1  # number of assets to include

        min_position_ratio = [0.5, 0.5, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.5]  # max position as a ratio of the bounds

        collision_mask = 1

        collapse_fixed_joints = True
        fix_base_link = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_mask_link_list = []  # For empty list, all links are labeled
        specific_filepath = None  # if not None, use this folder instead randomizing
        color = None

    class thin_asset_params(asset_state_params):
        num_assets = 0

        collision_mask = 1  # objects with the same collision mask will not collide

        max_position_ratio = [0.95, 0.95, 0.95]  # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.05]  # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0,
                              -1000.0]  # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [-np.pi, -np.pi, -np.pi]  # min euler angles
        max_euler_angles = [np.pi, np.pi, np.pi]  # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = True
        semantic_id = THIN_SEMANTIC_ID
        set_semantic_mask_per_link = False
        semantic_mask_link_list = []  ## If nothing is specified, all links are labeled
        color = [170, 66, 66]

    class tree_asset_params(asset_state_params):
        num_assets = 0

        collision_mask = 1  # objects with the same collision mask will not collide

        max_position_ratio = [0.95, 0.95, 0.1]  # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.0]  # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0,
                              -1000.0]  # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [0, -np.pi / 6.0, -np.pi]  # min euler angles
        max_euler_angles = [0, np.pi / 6.0, np.pi]  # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = True
        semantic_mask_link_list = []  ## If nothing is specified, all links are labeled
        semantic_id = TREE_SEMANTIC_ID
        color = [70, 200, 100]

    class object_asset_params(asset_state_params):
        num_assets = 1

        max_position_ratio = [0.95, 0.95, 0.95]  # min position as a ratio of the bounds
        min_position_ratio = [0.05, 0.05, 0.05]  # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0, -np.pi / 6, -np.pi]  # min euler angles
        max_euler_angles = [0, np.pi / 6, np.pi]  # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0]  # if > -900, use this value instead of randomizing

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = OBJECT_SEMANTIC_ID

        # color = [80,255,100]

    class wall1_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [2.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall2_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [7.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall3_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [12.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall4_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [17.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall5_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [22.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall6_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [27.5, -5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall7_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [2.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall8_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [7.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall9_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [12.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]


    class wall10_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [17.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall11_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [22.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]

    class wall12_map1(asset_state_params):
        num_assets = 1

        collision_mask = 1  # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 1.0, 0.5]  # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5]  # max position as a ratio of the bounds

        specified_position = [27.5, 5, 10]  # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0]  # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0]  # max euler angles

        specified_euler_angle = [0, 0, 0]  # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100, 200, 210]



    class asset_config:
        folder_path = f"{AERIAL_GYM_ROOT_DIR}/resources/models/environment_assets"

        include_asset_type = {
            "thin": False,
            "trees": False,
            "objects": False
        }

        include_env_bound_type = {
            "wall1_map1": False,
            "wall2_map1": False,
            "wall3_map1": False,
            "wall4_map1": False,
            "wall5_map1": False,
            "wall6_map1": False,
            "wall7_map1": False,
            "wall8_map1": False,
            "wall9_map1": False,
            "wall10_map1": False,
            "wall11_map1": False,
            "wall12_map1": False,
            # "wall7_map2": False,
            # "wall8_map2": False,
            #
            # "wall1_map1": False,
            # "wall1_map2": False,
            # "wall1_map3": False,
            # "wall1_map4": False,
            # "wall1_map5": False,
            # "wall1_map6": False,
            # "wall1_map7": False,
            # "wall2_map1": False,
            # "wall2_map2": False,
            # "wall2_map3": False,
            # "wall2_map4": False,
            # "wall2_map5": False,
            # "wall2_map6": False,
            # "wall2_map7": False,
            # "wall3_map1": False,
            # "wall3_map2": False,
            # "wall3_map3": False,
            # "wall3_map4": False,
            # "wall3_map5": False,
            # "wall3_map6": False,
            # "wall3_map7": False,
            # "wall4_map1": False,
            # "wall4_map2": False,
            # "wall4_map3": False,
            # "wall4_map4": False,
            # "wall4_map5": False,
            # "wall4_map6": False,
            # "wall4_map7": False,
            # "wall5_map1": False,
            # "wall5_map2": False,
            # "wall5_map3": False,
            # "wall5_map4": False,
            # "wall5_map5": False,
            # "wall5_map6": False,
            # "wall5_map7": False,
            # "wall6_map1": False,
            # "wall6_map2": False,
            # "wall6_map3": False,
            # "wall6_map4": False,
            # "wall6_map5": False,
            # "wall6_map6": False,
            # "wall6_map7": False,
            # "wall7_map1": False,
            # "wall7_map2": False,
            # "wall7_map3": False,
            # "wall7_map4": False,
            # "wall7_map5": False,
            # "wall7_map6": False,
            # "wall7_map7": False,
            # "wall8_map1": False,
            # "wall8_map2": False,
            # "wall8_map3": False,
            # "wall8_map4": False,
            # "wall8_map5": False,
            # "wall8_map6": False,
            # "wall8_map7": False,
        }

        env_lower_bound_min = [-5.0, -5.0, 0.0]  # lower bound for the environment space
        env_lower_bound_max = [-5.0, -5.0, 0.0]  # lower bound for the environment space
        env_upper_bound_min = [5.0, 5.0, 5.0]  # upper bound for the environment space
        env_upper_bound_max = [5.0, 5.0, 5.0]  # upper bound for the environment space
