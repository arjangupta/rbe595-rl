import random

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProject
from aeriel_robot_final_project import AerialRobotFinalProject

def main():

    map = random.randint(0,9)

    # asset_type_to_dict_map = {
    #     "thin": cfg.thin_asset_params,
    #     "trees": cfg.tree_asset_params,
    #     "objects": cfg.object_asset_params,
    #     "long_left_wall": cfg.left_wall,
    #     "long_right_wall": cfg.right_wall,
    #     "back_wall": cfg.back_wall,
    #     "front_wall": cfg.front_wall,
    #     "bottom_wall": cfg.bottom_wall,
    #     "top_wall": cfg.top_wall}

    cfg = AerialRobotCfgFinalProject(map)

    task_registry.register( "quad_for_final_project", AerialRobotFinalProject, cfg)
    env, env_cfg = task_registry.make_env("quad_for_final_project")
    # env, env_cfg = task_registry.make_env("quad_with_obstacles")

    print("Number of environments", env_cfg.env.num_envs)
    command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
    command_actions[:, 0] = 0.0
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    command_actions[:, 3] = 0.8

    env.enable_onboard_cameras = True

    for i in range(0, 5000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

if __name__ == "__main__":
    main()