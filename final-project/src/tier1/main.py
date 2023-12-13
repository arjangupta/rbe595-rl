import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProjectTier1
from aeriel_robot_final_project import AerialRobotFinalProjectTier1
from action_primitives import QuadActionPrimitives
from deep_q_learning import DeepQLearningAgent
from reward_system import QuadRewardSystem

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
        # Declare reward function
        self.reward_function = QuadRewardSystem()

    def step(self, action):
        # Get action bezier curve as sampled points
        num_samples = 5
        points = self.action_primitives.get_sampled_curve(action, num_samples=5)
        # Step through all points
        for i_sample in range(0, num_samples):
            # Set command actions
            self.command_actions = torch.from_numpy(points[:,i_sample])
            for _ in range(0, 50):
                # Step through the environment repeatedly
                obs, priviliged_obs, rewards, resets, extras = self.env.step(self.command_actions)
        return self.get_current_position(), self.reward_function.determine_reward(resets[0])

    def get_current_position(self):
        return self.env.get_current_position()[0]

def main():
    gym_iface = GymInterface()
    dql_agent = DeepQLearningAgent(gym_iface)
    dql_agent.train()

if __name__ == "__main__":
    main()