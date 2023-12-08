import sys
sys.path.insert(0, '../../../../workspace/aerial_gym_simulator')

import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry

env, env_cfg = task_registry.make_env("quad_with_obstacles")