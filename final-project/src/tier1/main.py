import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch
from aeriel_robot_cfg_final_project import AerialRobotCfgFinalProjectTier1
from aeriel_robot_final_project import AerialRobotFinalProjectTier1
from action_primitives import QuadActionPrimitives
from deep_q_learning import DeepQLearningAgent, State
from reward_system import QuadRewardSystem

class GymInterface:
    def __init__(self, debug=False):
        # Set debug
        self.debug = debug
        # Set trace
        self.trace = False
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
        # Set goal positions
        self.goal_position_options = torch.tensor(
            [[7.0, 6.5, 4.0],
             [5.0, -10.3, 6.4],
             [11.2, 2.2, 1.0],
             [3.0, -3.0, 3.0],
             [15.9, -10.0, 9.0],
             [2.0, 12.0, 7.1]
             ], dtype=torch.float32, device=self.device)
        self.goal_position = self.goal_position_options[0]
        # Get initial drone position
        self.initial_position = self.get_current_position().clone()
        if self.debug:
            print("Initial position: ", self.initial_position)
        # Set flag for first step done
        self.first_step_done = False
        # Current image array
        self.image_set = torch.zeros((3, 1024), dtype=torch.float32, device=self.device)
        # Moving setpoint time counter
        self.moving_setpoint_time_counter = 0.0
        self.moving_setpoint_time_counter_increment = 0.5

    def choose_new_goal_position(self):
        """Chooses a new goal position"""
        # Choose a random index from the goal position options
        # self.goal_position = self.goal_position_options[
        #     torch.randint(0, self.goal_position_options.shape[0], (1,)).item()]
        # Just choose a coordinate along the x-axis
        self.goal_position = torch.tensor([19.0, 0.0, 7.5],
                                          dtype=torch.float32, device=self.device)
        if self.debug:
            print("New goal position: ", self.goal_position)

    def step(self, action):
        # Capture starting relative position
        self.reward_function.dt_start = self.get_distance_from_moving_setpoint()
        if self.debug:
            print("dt_start: ", self.reward_function.dt_start)
        # Get action bezier curve as sampled points
        num_samples = 5  # THIS MUST NEVER BE LESS THAN 3 OR THE NN WILL FAIL
        points = self.action_primitives.get_sampled_curve(action, num_samples=5)
        # Step through all points
        reset = False
        collision = False
        for i_sample in range(0, num_samples):
            # Set command actions
            self.command_actions = torch.from_numpy(points[:, i_sample])
            for _ in range(0, 15):
                # Step through the environment repeatedly
                _, _, _, reset_ret, _, hit_ground_ret = self.env.step(self.command_actions)
                if reset_ret:
                    reset = True
                    self.moving_setpoint_time_counter = 0
                    if hit_ground_ret:
                        # print("Drone hit ground!")
                        collision = True
                    break
            # If one of the last 3 samples, save image
            if i_sample >= num_samples - 3:
                self.image_set[i_sample - num_samples + 3] = self.env.get_depth_image()
            if reset:
                break
        self.moving_setpoint_time_counter += self.moving_setpoint_time_counter_increment
        # Capture ending relative position
        self.reward_function.dt_end = self.get_distance_from_moving_setpoint()
        # Check if near goal
        near_goal = False
        if self.check_if_near_goal():
            near_goal = True
            print("Reached goal!")
        if self.debug:
            print("dt_end: ", self.reward_function.dt_end)
            print("Current position: ", self.get_current_position())
        return self.get_observation(), self.reward_function.determine_reward(collision,
                                                                             self.get_current_position()), reset, near_goal

    def get_observation(self):
        return State(self.image_set, self.get_relative_postion().unsqueeze(0))

    def get_image_set(self):
        return self.image_set

    def get_current_position(self):
        return self.env.get_current_position()[0]

    def calculate_3d_distance(self, A, B, C):
        """Returns the euclidean distance of the point C
        from the line AB.
            A, B and C are expected to be
        PyTorch tensors representing points
        in the form [x, y, z]"""
        AB = B - A
        AC = C - A
        cross_product = torch.cross(AB, AC)
        parallelogram_area = torch.norm(cross_product)
        length_AB = torch.norm(AB)  # which is the base of the parallelogram
        if self.trace:
            print("A: ", A)
            print("B: ", B)
            print("C: ", C)
            print("AB: ", AB)
            print("AC: ", AC)
            print("cross_product: ", cross_product)
            print("parallelogram_area: ", parallelogram_area)
            print("length_AB: ", length_AB)
        return parallelogram_area / length_AB  # which is the height of the parallelogram

    def get_distance_from_moving_setpoint(self):
        """The relative distance is the hypotenuse deviation of the drone
        from the moving setpoint which moves along the straight line path
        between the initial position and the goal position"""
        perpendicular_distance = self.calculate_3d_distance(
            self.initial_position,
            self.goal_position,
            self.get_current_position())
        # This is the ghost setpoint that moves along the straight line path. Its distance
        # is something that we incrementally increase as time goes on.
        distance_along_x_axis = self.moving_setpoint_time_counter - self.get_current_position()[0]
        # Now we use the Pythagorean theorem to calculate the relative distance
        relative_distance = torch.sqrt(perpendicular_distance**2 + distance_along_x_axis**2)
        if self.debug:
            print("relative_distance: ", relative_distance)
        return relative_distance

    def calculate_perpendicular_intersection(self, A, B, C):
        """Returns the point D which is the intersection of the
        perpendicular line from C to the line AB.
            A, B and C are PyTorch tensors
        representing points in the form [x, y, z]"""
        AB = B - A
        AC = C - A
        # Calculate the unit vector of AB
        AB_unit = AB / torch.norm(AB)
        # Project AC onto AB to get AD
        AD = torch.dot(AC, AB_unit) * AB_unit
        # Add AD to A to get the coordinates of D
        D = A + AD
        return D

    def get_relative_postion(self):
        """The relative position is the vector from the drone
        to the moving setpoint"""
        current_position = self.get_current_position()
        moving_setpoint = self.calculate_perpendicular_intersection(
            self.initial_position,
            self.goal_position,
            current_position)
        # Every "second" the setpoint moves a certain distance along the x-axis
        moving_setpoint[0] = moving_setpoint[0] + self.moving_setpoint_time_counter
        relative_pos = moving_setpoint  - current_position
        if self.debug:
            print("relative_pos: ", relative_pos)
        return relative_pos

    def check_if_near_goal(self):
        """Returns true if the drone is within 1 meter of the goal"""
        return torch.norm(self.goal_position - self.get_current_position()) < 1.0


def main():
    gym_iface = GymInterface(debug=False)
    dql_agent = DeepQLearningAgent(gym_iface)
    dql_agent.train()


if __name__ == "__main__":
    main()