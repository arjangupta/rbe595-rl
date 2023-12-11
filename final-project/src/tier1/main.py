import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProjectTier1
from aeriel_robot_final_project import AerialRobotFinalProjectTier1

import bezier
import numpy as np

def main():
    task_registry.register( "quad_for_final_project", AerialRobotFinalProjectTier1, AerialRobotCfgFinalProjectTier1())
    env, env_cfg = task_registry.make_env("quad_for_final_project")

    command_actions = torch.zeros((env_cfg.env.num_envs, 3))
    command_actions[:, 0] = 5.0
    command_actions[:, 1] = 5.0
    command_actions[:, 2] = 5.0
    # command_actions[:, 3] = 0.0

    for i in range(0, 5000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

def bezier_test():
    nodes = np.asfortranarray([
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
    ])
    curve = bezier.Curve(nodes, degree=2)
    s_vals = np.linspace(0.0, 1.0, 10)
    points = curve.evaluate_multi(s_vals)
    print(points)

if __name__ == "__main__":
    # main()
    bezier_test()