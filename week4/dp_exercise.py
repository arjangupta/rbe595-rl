# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning
# Week 4 Dynamic Programming Exercise

# Importing the necessary libraries
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


class Robot:
    """Class for taking action, returning reward"""

    def __init__(self, grid_world, goal_x=10, goal_y=7):
        """Constructor for the Robot class"""
        self.grid_world = grid_world
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.consider_occupied_spaces = True

    def get_reward(self, i, j):
        """Returns the reward for a given state and action"""
        # If the state is the goal state, return max reward
        if i == self.goal_y and j == self.goal_x:
            return 100
        # If the state is occupied, return min reward
        if self.grid_world[i, j] == 1:
            return -50
        # If the state is unoccupied, return -1
        else:
            return -1

    def take_action(self, i, j, action):
        """For actions 0-7, returns the new state after taking the action.
        The actions are enumerated in clockwise order, starting with 0 at 12 o'clock."""
        # If the action is 0 (up)
        if action == 0:
            # If the state is not in the top row
            if i != 0:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i - 1, j
                # If the state above is unoccupied
                elif self.grid_world[i - 1, j] == 0:
                    # # Return the new state
                    return i - 1, j
        # If the action is 1 (up-right)
        elif action == 1:
            # If the state is not in the top row or the rightmost column
            if i != 0 and j != self.grid_world.shape[1]-1:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i - 1, j + 1
                # If the state above and to the right is unoccupied
                elif self.grid_world[i - 1, j + 1] == 0:
                    # Return the new state
                    return i - 1, j + 1
        # If the action is 2 (right)
        elif action == 2:
            # If the state is not in the rightmost column
            if j != self.grid_world.shape[1]-1:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i, j+1
                # If the state to the right is unoccupied
                elif self.grid_world[i, j+1] == 0:
                    # Return the new state
                    return i, j+1
        # If the action is 3 (down-right)
        elif action == 3:
            # If the state is not in the bottom row or the rightmost column
            if i != self.grid_world.shape[0]-1 and j != self.grid_world.shape[1]-1:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i + 1, j + 1
                # If the state below and to the right is unoccupied
                elif self.grid_world[i + 1, j + 1] == 0:
                    # Return the new state
                    return i + 1, j + 1
        # If the action is 4 (down)
        elif action == 4:
            # If the state is not in the bottom row
            if i != self.grid_world.shape[0]-1:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i + 1, j
                # If the state below is unoccupied
                elif self.grid_world[i + 1, j] == 0:
                    # Return the new state
                    return i + 1, j
        # If the action is 5 (down-left)
        elif action == 5:
            # If the state is not in the bottom row or the leftmost column
            if i != self.grid_world.shape[0]-1 and j != 0:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i + 1, j - 1
                # If the state below and to the left is unoccupied
                elif self.grid_world[i + 1, j - 1] == 0:
                    # Return the new state
                    return i + 1, j - 1
        # If the action is 6 (left)
        elif action == 6:
            # If the state is not in the leftmost column
            if j != 0:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i, j - 1
                # If the state to the left is unoccupied
                elif self.grid_world[i, j - 1] == 0:
                    # Return the new state
                    return i, j - 1
        # If the action is 7 (up-left)
        elif action == 7:
            # If the state is not in the top row or the leftmost column
            if i != 0 and j != 0:
                if self.consider_occupied_spaces:
                    # Return the new state
                    return i - 1, j - 1
                # If the state above and to the left is unoccupied
                elif self.grid_world[i - 1, j - 1] == 0:
                    # Return the new state
                    return i - 1, j - 1
        # If the action is invalid, return the current state
        return i, j

    def get_stochastic_action_rewards(self, current_action, i, j):
        """Returns the rewards for +/-45 degree actions"""
        # Get actions that are +/-45 degrees current action
        action_plus_45 = (current_action + 1) % 8
        action_minus_45 = (current_action - 1) % 8
        # Take action
        i_plus_45, j_plus_45 = self.take_action(i, j, action_plus_45)
        i_minus_45, j_minus_45 = self.take_action(i, j, action_minus_45)
        # Calculate the reward for the action
        reward_plus_45 = self.get_reward(i_plus_45, j_plus_45)
        reward_minus_45 = self.get_reward(i_minus_45, j_minus_45)
        # Return rewards
        return reward_plus_45, i_plus_45, j_plus_45, reward_minus_45, i_minus_45, j_minus_45

class PolicyIteration:
    """Class for the Policy Iteration algorithm as described in the Barto & Sutton textbook"""

    def __init__(self, probability, grid_world, goal_x=10, goal_y=7, gamma=0.95, theta=0.01, generalized=False):
        """Constructor for the Policy Iteration algorithm"""
        self.probability = probability # p(s',r|s,a), chance that if robot attemts to do action a from state s it will go to s' and get expected reward
        self.grid_world = grid_world
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.gamma = gamma
        self.theta = theta
        self.generalized = generalized
        # Initialize a reward system
        self.robot = Robot(self.grid_world, self.goal_x, self.goal_y)
        # Initialize a value function of random nonzero real numbers
        # self.value_function = np.random.uniform(-1, 1, self.grid_world.shape)
        self.value_function = np.zeros(self.grid_world.shape)
        # Initialize a policy of random actions for each state (0-7)
        self.policy = np.random.randint(0, 8, self.grid_world.shape)

    def policy_evaluation(self):
        """Performs policy evaluation step of algorithm"""
        # Repeat until delta < theta
        while True:
            # Initialize delta to 0
            delta = 0
            # For each state in the grid world
            for i in range(self.grid_world.shape[0]):
                for j in range(self.grid_world.shape[1]):
                    # if self.grid_world[i, j] == 0:
                    # Calculate the value function for the state
                    v = self.value_function[i,j]
                    # Update the value function
                    self.value_function[i, j] = self.calculate_value_function(i, j)
                    # Calculate the difference between the old value function and the new value function
                    delta = max(delta, abs(v - self.value_function[i, j]))
            # avg_delta = delta/(self.grid_world.shape[0]*self.grid_world.shape[1])
            print("Delta: ", delta)
            # print("Value Function: ", self.value_function)
            # If delta < theta, then break
            if delta < self.theta:
                break
            if self.generalized:
                break

    def calculate_value_function_bellman(self, i, j):
        value_summation = 0

        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            if self.robot.consider_occupied_spaces and new_i==i and new_j==j:
                continue
            # Calculate the reward for the action
            reward = self.robot.get_reward(new_i, new_j)
            # Calculate total value summation
            value_summation += .125 * self.probability[i,j] * (reward + self.gamma * self.value_function[new_i, new_j])

        return value_summation

    def calculate_value_function(self, i, j):
        """Calculates the value function for a state"""
        # Summation variable
        value_summation = 0
        # Get action from policy
        action = self.policy[i, j]
        # Take action
        new_i, new_j = self.robot.take_action(i, j, action)
        # if self.robot.consider_occupied_spaces and new_i == i and new_j == j:
        #     print("new i and j are the same as old i and j")
        #     return -100
        # Calculate the reward for the action
        reward = self.robot.get_reward(new_i, new_j)
        # Add to total value summation
        value_summation += self.probability[i,j] * (reward + self.gamma * self.value_function[new_i, new_j])
        if (1 - self.probability[i,j]) > 0:
            # Get actions that are +/-45 degrees current action
            stochastic_rewards = self.robot.get_stochastic_action_rewards(action, i, j)
            reward_plus_45, i_plus_45, j_plus_45, reward_minus_45, i_minus_45, j_minus_45 = stochastic_rewards
            # Add to total value summation
            minority_prob = 1 - self.probability[i,j]
            value_summation += minority_prob/2 * (reward_plus_45 + self.gamma * self.value_function[i_plus_45, j_plus_45])
            value_summation += minority_prob/2 * (reward_minus_45 + self.gamma * self.value_function[i_minus_45, j_minus_45])
        # Print the value summation
        # print("Value Summation: ", value_summation)
        return value_summation

    def policy_improvement(self):
        """Performs policy improvement step of algorithm"""
        # Initialize a boolean flag to false
        #FIXME: should we do a count for this instead? make sure all values are stable?
        policy_stable = True
        # For each state in the grid world
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                # If the state is unoccupied
                # if self.grid_world[i, j] == 0:
                # Store the old action
                old_action = self.policy[i, j]
                # Calculate the new action
                new_action = self.calculate_new_action(i, j)
                # Update the policy
                self.policy[i, j] = new_action
                # If the old action and the new action are the same, set the flag to true
                if old_action != new_action:
                    policy_stable = False
        # Return the policy and the boolean flag
        return policy_stable            

    def calculate_new_action(self, i, j):
        """Calculates the new action for a state"""
        # Initialize a list of action values
        pi_values = []
        # For each action
        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            # if self.robot.consider_occupied_spaces and new_i == i and new_j == j:
            #     pi_values.append(0)
            # Calculate the reward for the action
            reward = self.robot.get_reward(new_i, new_j)
            # Calculate the action value
            action_value = self.probability[i, j] * (reward + self.gamma * self.value_function[new_i, new_j])
            if (1 - self.probability[i, j]) > 0:
                # Get actions that are +/-45 degrees current action
                stochastic_rewards = self.robot.get_stochastic_action_rewards(action, i, j)
                reward_plus_45, i_plus_45, j_plus_45, reward_minus_45, i_minus_45, j_minus_45 = stochastic_rewards
                # Add to total value summation
                minority_prob = 1 - self.probability[i, j]
                action_value += minority_prob/2 * (reward_plus_45 + self.gamma * self.value_function[i_plus_45, j_plus_45])
                action_value += minority_prob/2 * (reward_minus_45 + self.gamma * self.value_function[i_minus_45, j_minus_45])
            # Add to list of action values
            pi_values.append(action_value)
        # Return the action with the maximum action value
        return np.argmax(pi_values)
    
    def run(self):
        """Run the policy iteration algorithm as described in Page 80 of textbook"""
        policy_stable = False
        i = 0
        while not policy_stable:
            # For every N iterations, print the value function and policy
            if i % 500 == 0:
                print("Iteration: ", i)
                print("Policy: ", self.policy)
                print("Value Function: ", self.value_function)
                print()
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            i += 1
        print("Policy: ", self.policy)
        print("Value Function: ", self.value_function)
        return self.value_function, self.policy

class ValueIteration:
    """Class for the Value Iteration algorithm as described in the Barto & Sutton textbook"""

    def __init__(self, probability, grid_world, goal_x=10, goal_y=7, gamma=0.95, theta=0.01):
        """Constructor for the Value Iteration algorithm"""
        self.probability = probability  # p(s',r|s,a), chance that if robot attemts to do action a from state s it will go to s' and get expected reward
        self.grid_world = grid_world
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.gamma = gamma
        self.theta = theta
        # Initialize a reward system
        self.robot = Robot(self.grid_world, self.goal_x, self.goal_y)
        # Initialize a value function of random nonzero real numbers
        # self.value_function = np.random.uniform(-1, 1, self.grid_world.shape)
        self.value_function = np.zeros(self.grid_world.shape)
        # Initialize a policy of random actions for each state (0-7)
        self.policy = np.random.randint(0, 8, self.grid_world.shape)

    def main_loop(self):
        """Performs main-loop step of algorithm"""
        # Repeat until delta < theta
        while True:
            # Initialize delta to 0
            delta = 0
            # For each state in the grid world
            for i in range(self.grid_world.shape[0]):
                for j in range(self.grid_world.shape[1]):
                    # if self.grid_world[i, j] == 0:
                    # Calculate the value function for the state
                    v = self.value_function[i, j]
                    # Update the value function
                    self.value_function[i, j] = self.get_max_action(i, j)
                    # Calculate the difference between the old value function and the new value function
                    delta = max(delta, abs(v - self.value_function[i, j]))
            print("Delta: ", delta)
            # If delta < theta, then break
            if delta < self.theta:
                break

    def get_max_action(self, i, j):
        leaf_values = []
        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            # Calculate the reward for the action
            reward = self.robot.get_reward(new_i, new_j)
            # Calculate total value summation
            value_summation = self.probability[i,j] * (reward + self.gamma * self.value_function[new_i, new_j])
            if (1 - self.probability[i,j]) > 0:
                # Get actions that are +/-45 degrees current action
                stochastic_rewards = self.robot.get_stochastic_action_rewards(action, i, j)
                reward_plus_45, i_plus_45, j_plus_45, reward_minus_45, i_minus_45, j_minus_45 = stochastic_rewards
                # Add to total value summation
                minority_prob = 1 - self.probability[i,j]
                value_summation += minority_prob/2 * (reward_plus_45 + self.gamma * self.value_function[i_plus_45, j_plus_45])
                value_summation += minority_prob/2 * (reward_minus_45 + self.gamma * self.value_function[i_minus_45, j_minus_45])
            leaf_values.append(value_summation)
        return np.max(leaf_values)

    def policy_improvement(self):
        """Performs policy improvement step of algorithm"""
        # Initialize a boolean flag to false
        policy_stable = False
        # For each state in the grid world
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                self.policy[i, j] = self.find_arg_max_action(i, j)

    def find_arg_max_action(self, i, j):
        """Calculates the new action for a state"""
        # Initialize a list of action values
        action_values = []
        # For each action
        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            # if self.robot.consider_occupied_spaces and new_i == i and new_j == j:
            #     action_values.append(0)
            # Calculate the reward for the action
            reward = self.robot.get_reward(new_i, new_j)
            # Calculate the action value
            action_value = 0.125 *  self.probability[i,j] * (reward + self.gamma * self.value_function[new_i, new_j])
            if (1 - self.probability[i,j]) > 0:
                # Get actions that are +/-45 degrees current action
                stochastic_rewards = self.robot.get_stochastic_action_rewards(action, i, j)
                reward_plus_45, i_plus_45, j_plus_45, reward_minus_45, i_minus_45, j_minus_45 = stochastic_rewards
                # Add to total value summation
                minority_prob = 1 - self.probability[i,j]
                action_value += .125 * minority_prob/2 * (reward_plus_45 + self.gamma * self.value_function[i_plus_45, j_plus_45])
                action_value += .125 * minority_prob/2 * (reward_minus_45 + self.gamma * self.value_function[i_minus_45, j_minus_45])
            # Add to list of action values
            action_values.append(action_value)
        # Return the action with the maximum action value
        return np.argmax(action_values)

    def run(self):
        """Run the value iteration algorithm as described in Page 83 of textbook"""

        print("Policy: ", self.policy)
        print("Value Function: ", self.value_function)

        self.main_loop()
        self.policy_improvement()

        print("Policy: ", self.policy)
        print("Value Function: ", self.value_function)

        return self.value_function, self.policy

def plot_2d_array_with_arrows(gridworld, policy, goal_y=7, goal_x=10):
    """Takes in a 2D array of 0's and 1's and converts
    it to a plot of occupied and unoccupied spaces, with arrows"""

    # Creating a figure and axes
    fig, ax = plt.subplots()
    # Set the size of the figure
    fig.set_size_inches(14, 7)
    # Creating a plot of the array
    ax.imshow(gridworld, cmap='binary')
    # Color the goal state red
    ax.plot(goal_x, goal_y, 'ro')
    # Form the mesh grid
    X, Y = np.meshgrid(np.arange(gridworld.shape[1]), np.arange(gridworld.shape[0]))
    U, V = create_arrows(policy, gridworld)
    # Plot the arrows
    ax.quiver(X, Y, U, V)
    # Show ticks at every integer
    ax.set_xticks(np.arange(0, gridworld.shape[1], 1))
    ax.set_yticks(np.arange(0, gridworld.shape[0], 1))
    # Decrease text size along the axes
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Displaying the plot
    plt.show()

def create_arrows(policy, gridworld):
    U = np.zeros(policy.shape)
    V = np.zeros(policy.shape)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            # Try to hide arrows in the occupied spaces
            if gridworld[i, j] == 1:
                U[i,j] = 1
                V[i,j] = 0.1
            else:
                # If the action is 0 (up)
                if policy[i,j]==0:
                    U[i,j]=0
                    V[i,j]=1

                # If the action is 1 (up-right)
                if policy[i,j]==1:
                    U[i,j]=np.cos(math.pi/4)
                    V[i,j]=np.sin(math.pi/4)

                # If the action is 2 (right)
                if policy[i,j]==2:
                    U[i,j]=1
                    V[i,j]=0

                # If the action is 3 (down-right)
                if policy[i,j]==3:
                    U[i, j] = np.cos(-1*math.pi / 4)
                    V[i, j] = np.sin(-1*math.pi / 4)

                # If the action is 4 (down)
                if policy[i,j]==4:
                    U[i,j]=0
                    V[i,j]=-1

                # If the action is 5 (down-left)
                if policy[i,j]==5:
                    U[i, j] = np.cos(-3 * math.pi / 4)
                    V[i, j] = np.sin(-3 * math.pi / 4)

                # If the action is 6 (left)
                if policy[i,j]==6:
                    U[i,j]=-1
                    V[i,j]=0

                # If the action is 7 (up-left)
                if policy[i,j]==7:
                    U[i, j] = np.cos(3 * math.pi / 4)
                    V[i, j] = np.sin(3 * math.pi / 4)

    # If i/j is in leftmost or rightmost column, set the action as 4 (down)
    # If i/j is in the topmost or bottommost row, set the action as 2 (right)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            if j==0:
                U[i,j]=-1
                V[i,j]=0
            if j==policy.shape[1]-1:
                U[i,j]=1
                V[i,j]=0
            if i==0:
                U[i,j]=0
                V[i,j]=1
            if i==policy.shape[0]-1:
                U[i,j]=0
                V[i,j]=-1

    return U,V

def plot_2d_array_with_grid(gridworld, values, goal_y=7, goal_x=10):
    """Takes in a 2D array of 0's and 1's and converts
    it to a plot of occupied and unoccupied spaces, with a grid for every cell"""

    # Creating a figure and axes
    fig, ax = plt.subplots()
    # Set the size of the figure
    fig.set_size_inches(14, 7)
    # Creating a plot of the array
    ax.imshow(gridworld, cmap='binary')
    # Color the goal state red
    ax.plot(goal_x, goal_y, 'ro')
    # Form the grid lines such that they are in the middle of each cell
    ax.set_xticks(np.arange(-.5, gridworld.shape[1], 1))
    ax.set_yticks(np.arange(-.5, gridworld.shape[0], 1))
    # Hide the tick marks
    ax.tick_params(axis='both', which='both', length=0)
    # Decrease text size along the axes
    ax.tick_params(axis='both', which='major', labelsize=6)
    # Display the grid
    ax.grid()
    # Normalize the values
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
    # Show a section of the normalized values
    print("first 5x5 of normalized values: ")
    print(normalized[0:5, 0:5])
    # For every unoccupied cell, fill (add_patch) it in as a shade of grey of the normalized value
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if gridworld[i, j] == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=(normalized[i,j], normalized[i,j], normalized[i,j])))
    # Displaying the plot
    plt.show()

# Defining the main function
def main(model_type, alg_type):
    # Creating a 2D array of 0,s and 1,
    grid_world = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    if model_type == "Deterministic":
        probability = np.ones(grid_world.shape)
    else:
        probability = np.full(grid_world.shape, .8)

    if alg_type == "PolicyIteration":
        # Run Policy Iteration algorithm
        policy_iteration = PolicyIteration(probability, grid_world, generalized=False, theta=0.01)
        print("Running Policy Iteration algorithm...")
        value_function, policy = policy_iteration.run()
    elif alg_type == "ValueIteration":
        # Run Value Iteration
        value_iteration = ValueIteration(probability, grid_world, theta=0.000000000000000000000099)
        print("Running Value Iteration algorithm...")
        value_function, policy = value_iteration.run()
    else:
        # Run Generalized Policy Iteration
        policy_iteration = PolicyIteration(probability, grid_world, generalized=True, theta=0.01)
        print("Running Generalized Policy Iteration algorithm...")
        value_function, policy = policy_iteration.run()
    plot_2d_array_with_arrows(grid_world, policy)
    plot_2d_array_with_grid(grid_world, value_function)

# Calling the main function
if __name__ == "__main__":
    if len(sys.argv) <= 2 or \
            (sys.argv[1] != "Deterministic" and sys.argv[1] != "Stochastic") or \
            (sys.argv[2] != "PolicyIteration" and sys.argv[2] != "ValueIteration" and sys.argv[2] != "GeneralizedPolicyIteration"):
        print("Please enter either \"Deterministic\" or \"Stochastic\" for your first argument, and either \"PolicyIteration\", \"ValueIteration\", or \"GeneralizedPolicyIteration\" for your second")
    else:
        main(sys.argv[1], sys.argv[2])