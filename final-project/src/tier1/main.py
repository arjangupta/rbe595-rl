import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProjectTier1
from aeriel_robot_final_project import AerialRobotFinalProjectTier1
from action_primitives import QuadActionPrimitives

class GymInterface:
    def __init__(self):
        # Declare environment
        task_registry.register( "quad_for_final_project", AerialRobotFinalProjectTier1, AerialRobotCfgFinalProjectTier1())
        # Create environment
        self.env, self.env_cfg = task_registry.make_env("quad_for_final_project")
        # Declare command actions
        self.command_actions = torch.zeros((self.env_cfg.env.num_envs, 3))
        # Declare action primitives
        self.action_primitives = QuadActionPrimitives()

    def step(self, i):
        # Cycle through all actions
        action = i % 18
        print("action: " + str(action))
        # Get sampled curve
        points = self.action_primitives.get_sampled_curve(action, num_samples=5)

        # Step through all points
        for j in range(0, 5):
            # Set command actions
            self.command_actions = torch.from_numpy(points[:,j])
            if i % 250 == 0:
                print(f"command_actions: {self.command_actions}")
            for k in range(0, 50):
                if j == 4 and k == 49:
                    curr_pos = self.env.get_current_position()
                    print(f"Reached target pos of action: {curr_pos}")
                # Step through the environment repeatedly
                obs, priviliged_obs, rewards, resets, extras = self.env.step(self.command_actions)

def main():
    gym_iface = GymInterface()

    for i in range(0, 5000):
        gym_iface.step(i)

if __name__ == "__main__":
    main()