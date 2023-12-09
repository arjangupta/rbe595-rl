import env_var_loader
AERIEL_GYM_PATH = env_var_loader.get_aerial_gym_path()

import sys
sys.path.insert(0, AERIEL_GYM_PATH)

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProject
from aeriel_robot_final_project import AerialRobotFinalProject

def main():
    task_registry.register( "quad_for_final_project", AerialRobotFinalProject, AerialRobotCfgFinalProject())
    env, env_cfg = task_registry.make_env("quad_for_final_project")

    print("Number of environments", env_cfg.env.num_envs)
    command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
    command_actions[:, 0] = 0.0
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    command_actions[:, 3] = 0.8

    env.enable_onboard_cameras = True

    for i in range(0, 5000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
            
        print("Done", i)

if __name__ == "__main__":
    main()