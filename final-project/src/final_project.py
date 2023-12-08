import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AERIEL_GYM_PATH = os.environ.get("AERIEL_GYM_PATH")

if AERIEL_GYM_PATH is None:
    raise Exception("Please set the AERIEL_GYM_PATH environment variable")
else:
    print("AERIEL_GYM_PATH set as:", AERIEL_GYM_PATH)

import sys
sys.path.insert(0, AERIEL_GYM_PATH)

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

env, env_cfg = task_registry.make_env("quad_with_obstacles")

print("Number of environments", env_cfg.env.num_envs)
command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
command_actions[:, 0] = 0.0
command_actions[:, 1] = 0.0
command_actions[:, 2] = 0.0
command_actions[:, 3] = 0.8

for i in range(0, 500):
    obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
        
    print("Done", i)