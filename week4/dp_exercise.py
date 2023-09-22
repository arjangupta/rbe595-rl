# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning
# Week 4 Dynamic Programming Exercise

# Importing the necessary libraries
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

    def get_reward(self, i, j, action):
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
                # If the state above is unoccupied
                if self.grid_world[i-1, j] == 0:
                    # Return the new state
                    return i-1, j
        # If the action is 1 (up-right)
        elif action == 1:
            # If the state is not in the top row or the rightmost column
            if i != 0 and j != self.grid_world.shape[1]-1:
                # If the state above and to the right is unoccupied
                if self.grid_world[i-1, j+1] == 0:
                    # Return the new state
                    return i-1, j+1
        # If the action is 2 (right)
        elif action == 2:
            # If the state is not in the rightmost column
            if j != self.grid_world.shape[1]-1:
                # If the state to the right is unoccupied
                if self.grid_world[i, j+1] == 0:
                    # Return the new state
                    return i, j+1
        # If the action is 3 (down-right)
        elif action == 3:
            # If the state is not in the bottom row or the rightmost column
            if i != self.grid_world.shape[0]-1 and j != self.grid_world.shape[1]-1:
                # If the state below and to the right is unoccupied
                if self.grid_world[i+1, j+1] == 0:
                    # Return the new state
                    return i+1, j+1
        # If the action is 4 (down)
        elif action == 4:
            # If the state is not in the bottom row
            if i != self.grid_world.shape[0]-1:
                # If the state below is unoccupied
                if self.grid_world[i+1, j] == 0:
                    # Return the new state
                    return i+1, j
        # If the action is 5 (down-left)
        elif action == 5:
            # If the state is not in the bottom row or the leftmost column
            if i != self.grid_world.shape[0]-1 and j != 0:
                # If the state below and to the left is unoccupied
                if self.grid_world[i+1, j-1] == 0:
                    # Return the new state
                    return i+1, j-1
        # If the action is 6 (left)
        elif action == 6:
            # If the state is not in the leftmost column
            if j != 0:
                # If the state to the left is unoccupied
                if self.grid_world[i, j-1] == 0:
                    # Return the new state
                    return i, j-1
        # If the action is 7 (up-left)
        elif action == 7:
            # If the state is not in the top row or the leftmost column
            if i != 0 and j != 0:
                # If the state above and to the left is unoccupied
                if self.grid_world[i-1, j-1] == 0:
                    # Return the new state
                    return i-1, j-1
        # If the action is invalid, return the current state
        return i, j

class PolicyIteration:
    """Class for the Policy Iteration algorithm as described in the Barto & Sutton textbook"""

    def __init__(self, probability, grid_world, goal_x=10, goal_y=7, gamma=0.95, theta=0.01, generalized=False):
        """Constructor for the Policy Iteration algorithm"""
        self.probability = probability
        self.grid_world = grid_world
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.gamma = gamma
        self.theta = theta
        self.generalized = generalized
        # Initialize a reward system
        self.robot = Robot(self.grid_world, self.goal_x, self.goal_y)
        # Initialize a value function of zeros
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
                    # If the state is unoccupied
                    if self.grid_world[i, j] == 0:
                        # Calculate the value function for the state
                        v = self.calculate_value_function(i, j)
                        # Calculate the difference between the old value function and the new value function
                        delta = max(delta, abs(self.value_function[i, j] - v))
                        # Update the value function
                        self.value_function[i, j] = v
            # If delta < theta, then break
            if delta < self.theta:
                break
            if self.generalized:
                break

    def policy_improvement(self):
        """Performs policy improvement step of algorithm"""
        # Initialize a boolean flag to false
        policy_stable = False
        # For each state in the grid world
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                # If the state is unoccupied
                if self.grid_world[i, j] == 0:
                    # Store the old action
                    old_action = self.policy[i, j]
                    # Calculate the new action
                    new_action = self.calculate_new_action(i, j)
                    # Update the policy
                    self.policy[i, j] = new_action
                    # If the old action and the new action are the same, set the flag to true
                    if old_action == new_action:
                        policy_stable = True
        # Return the policy and the boolean flag
        return policy_stable

    def calculate_value_function(self, i, j):
        """Calculates the value function for a state"""
        # Summation variable
        value_summation = 0
        # For each action
        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            # Calculate the reward for the action
            reward = self.robot.get_reward(i, j, action)
            # Add to total value summation
            #FIXME: which one?
            value_summation += self.policy[i, j] * self.probability[i, j] * (reward + self.gamma * self.value_function[new_i, new_j])
            value_summation += reward + self.gamma * self.value_function[new_i, new_j]
        # Return the value summation
        return value_summation
            

    def calculate_new_action(self, i, j):
        """Calculates the new action for a state"""
        # Initialize a list of action values
        action_values = []
        # For each action
        for action in range(8):
            # Take action
            new_i, new_j = self.robot.take_action(i, j, action)
            # Calculate the reward for the action
            reward = self.robot.get_reward(i, j, action)
            # Calculate the action value
            action_value = reward + self.gamma * self.value_function[new_i, new_j]
            # Add to list of action values
            action_values.append(action_value)
        # Return the action with the maximum action value
        return np.argmax(action_values)
    
    def run(self):
        """Run the policy iteration algorithm as described in Page 80 of textbook"""
        keep_running = True
        i = 0
        while keep_running:
            # For every N iterations, print the value function and policy
            if i % 500 == 0:
                print("Iteration: ", i)
                print("Value Function: ", self.value_function)
                print("Policy: ", self.policy)
                print()
            self.policy_evaluation()
            keep_running = self.policy_improvement()
            i += 1

        return self.value_function, self.policy


def plot_2d_array_with_arrows(array, policy, goal_y=7, goal_x=10):
    """Takes in a 2D array of 0's and 1's and converts
    it to a plot of occupied and unoccupied spaces, with arrows"""

    # Creating a figure and axes
    fig, ax = plt.subplots()
    # Set the size of the figure
    fig.set_size_inches(14, 7)
    # Creating a plot of the array
    ax.imshow(array, cmap='binary')
    # Color the goal state red
    ax.plot(goal_x, goal_y, 'ro')
    # Form the mesh grid
    X, Y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    U = np.cos(policy)
    V = np.sin(policy)
    # Try to hide arrows in the occupied spaces
    U[array == 1] = 1
    V[array == 1] = 0.1
    # Plot the arrows
    ax.quiver(X, Y, U, V)
    # Print X, Y, U, V for only the first 5 rows and columns
    print("X first5: ", X[:5, :5])
    print("Y first5: ", Y[:5, :5])
    print("U first5: ", U[:5, :5])
    print("V first5: ", V[:5, :5])
    # Show ticks at every integer
    ax.set_xticks(np.arange(0, array.shape[1], 1))
    ax.set_yticks(np.arange(0, array.shape[0], 1))
    # Decrease text size along the axes
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Displaying the plot
    plt.show()


def plot_2d_array_with_grid(array, goal_y=7, goal_x=10):
    """Takes in a 2D array of 0's and 1's and converts
    it to a plot of occupied and unoccupied spaces, with a grid for every cell"""

    # Creating a figure and axes
    fig, ax = plt.subplots()
    # Set the size of the figure
    fig.set_size_inches(14, 7)
    # Creating a plot of the array
    ax.imshow(array, cmap='binary')
    # Color the goal state red
    ax.plot(goal_x, goal_y, 'ro')
    # Form the grid lines such that they are in the middle of each cell
    ax.set_xticks(np.arange(-.5, array.shape[1], 1))
    ax.set_yticks(np.arange(-.5, array.shape[0], 1))
    # Hide the tick marks
    ax.tick_params(axis='both', which='both', length=0)
    # Decrease text size along the axes
    ax.tick_params(axis='both', which='major', labelsize=6)
    # Display the grid
    ax.grid()
    # For every unoccupied cell, fill it in as a random shade of grey
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                grey = np.random.uniform(0.4, 0.9)
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=(grey, grey, grey)))
    # Displaying the plot
    plt.show()


# Defining the main function
def main(model_type):
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

    # Run Policy Iteration algorithm
    policy_iteration = PolicyIteration(probability, grid_world, generalized=False, theta=5)
    print("Running Policy Iteration algorithm...")
    value_function, policy = policy_iteration.run()

# Calling the main function
if __name__ == "__main__":
    if len(sys.argv) <= 1 or (sys.argv[1] != "Deterministic" and sys.argv[1] != "Stochastic"):
        print("Please enter either \"Deterministic\" or \"Stochastic\"")
    else:
        main(sys.argv[1])