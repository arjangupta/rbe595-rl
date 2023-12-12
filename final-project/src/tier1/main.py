import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProjectTier1
from aeriel_robot_final_project import AerialRobotFinalProjectTier1
from action_primitives import QuadActionPrimitives

def main():
    task_registry.register( "quad_for_final_project", AerialRobotFinalProjectTier1, AerialRobotCfgFinalProjectTier1())
    env, env_cfg = task_registry.make_env("quad_for_final_project")

    command_actions = torch.zeros((env_cfg.env.num_envs, 3))
    # command_actions[:, 0] = 5.0
    # command_actions[:, 1] = 5.0
    # command_actions[:, 2] = 5.0
    # # command_actions[:, 3] = 0.0

    # Declare action primitives
    action_primitives = QuadActionPrimitives()

    for i in range(0, 5000):
        # Cycle through all actions
        action = i % 18
        print("action: " + str(action))
        # Get sampled curve
        points = action_primitives.get_sampled_curve(action, num_samples=5)

        # Step through all points
        for j in range(0, 5):
            # Set command actions
            command_actions = torch.from_numpy(points[:,j])
            if i % 250 == 0:
                print(f"command_actions: {command_actions}")
            for k in range(0, 50):
                if j == 4 and k == 49:
                    curr_pos = env.get_current_position()
                    print(f"Reached target pos of action: {curr_pos}")
                # Step through the environment repeatedly
                obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)

if __name__ == "__main__":
    main()