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
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set goal position
        self.env.goal_position = torch.tensor([7.0, 7.0, 2.0], dtype=torch.float32, device=self.device)
        # Get initial drone position
        self.initial_position = self.get_current_position()

    def step(self, action):
        # Capture starting relative position
        self.reward_function.dt_start = self.get_relative_position()
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
        # Capture ending relative position
        self.reward_function.dt_end = self.get_relative_position()
        return self.get_current_position(), self.reward_function.determine_reward(False)

    def get_current_position(self):
        return self.env.get_current_position()[0]
    
    def calculate_3d_distance(A, B, C):
        # A, B and C are expected to be
        # PyTorch tensors representing points
        # in the form [x, y, z]
        AB = B - A
        AC = C - A
        cross_product = torch.cross(AB, AC)
        numerator = torch.norm(cross_product)
        denominator = torch.norm(AB)

        return numerator / denominator

    def calculate_perpendicular_intersection(A, B, C):
        # A, B and C are expected to be
        # PyTorch tensors representing points
        # in the form [x, y, z]
        AB = B - A
        AC = C - A
        # Calculate the unit vector of AB
        AB_unit = AB / torch.norm(AB)
        # Project AC onto AB to get AD
        AD = torch.dot(AC, AB_unit) * AB_unit
        # Add AD to A to get the coordinates of D
        D = A + AD
        return D

    def get_relative_position(self):
        # The relative position is the deviation from the straight line path
        # between the initial position and the goal position
        return self.calculate_deviation_3d(
            self.initial_position,
            self.env.goal_position,
            self.get_current_position())

def main():
    gym_iface = GymInterface()
    dql_agent = DeepQLearningAgent(gym_iface)
    dql_agent.train()

if __name__ == "__main__":
    main()