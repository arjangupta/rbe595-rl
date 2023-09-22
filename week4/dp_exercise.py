# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning
# Week 4 Dynamic Programming Exercise

# Importing the necessary libraries
import sys

import matplotlib.pyplot as plt
import numpy as np


class PolicyIteration:
    """Class for the Policy Iteration algorithm as described in the Barto & Sutton textbook"""

    def __init__(self, policy, probability, grid_world, goal_x=10, goal_y=7, gamma=0.95, theta=0.01):
        """Constructor for the Policy Iteration algorithm"""
        self.policy = policy
        self.probability = probability
        self.grid_world = grid_world
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.gamma = gamma
        self.theta = theta

    def policy_evaluation(self):
        """Performs policy evaluation on a policy"""
        # Initialize a value function of zeros
        value_function = np.zeros(self.grid_world.shape)
        # Repeat until delta < theta
        while True:
            # Initialize delta to 0
            delta = 0
            # For each state in the grid world
            for i in range(self.grid_world.shape[0]):
                for j in range(self.grid_world.shape[1]):
                    # If the state is unoccupied
                    if self.grid_world[i, j] == 0:  # FIXME: do we need this?
                        # Calculate the value function for the state
                        v = self.calculate_value_function(i, j)
                        # Calculate the difference between the old value function and the new value function
                        delta = max(delta, abs(value_function[i, j] - v))
                        # Update the value function
                        value_function[i, j] = v
            # If delta < theta, then break
            if delta < self.theta:
                break
        # Return the value function
        return value_function

    def policy_improvement(self, value_function):
        """Performs policy improvement on a policy"""
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
                    new_action = self.calculate_new_action(i, j, value_function)
                    # Update the policy
                    self.policy[i, j] = new_action
                    # If the old action and the new action are the same, set the flag to true
                    if old_action == new_action:
                        policy_stable = True
        # Return the policy and the boolean flag
        return self.policy, policy_stable

    def calculate_value_function(self, i, j):
        """Calculates the value function for a state"""
        # Calculate the value function for the state
        v = 0
        # If the state is not the goal state
        if i != self.goal_y and j != self.goal_x:
            # Calculate the value function for the state
            # FIXME: why cosine???
            # FIXME: we should be using the value of all the states around it right???
            eight_connected_locations = self.calculate_eight_connected(i, j)
            v = 0
            for neighbor in eight_connected_locations:
                v += self.policy[i,j] * self.probability[i, j] * (self.reward[i,j] + self.gamma * self.calculate_value_function_for_action(neighbor[0],neighbor[1],self.policy[neighbor[0],neighbor[1]]))
        # Return the value function
        return v

    def calculate_eight_connected(self, row, col):
        locations = list()
        for i in range(-1, 1, 1):
            if 0 <= row + i < len(self.grid_world[0]):
                for j in range(-1, 1, 1):
                    if 0 <= col + j < len(self.grid_world[1]):
                        locations.append((row + i, col + j))
        return locations

    def calculate_new_action(self, i, j, value_function):
        """Calculates the new action for a state"""
        # Initialize an array to store the value function for each action
        value_function_for_action = np.zeros(4)
        # For each action
        for action in range(4):
            # Calculate the value function for the action
            value_function_for_action[action] = self.calculate_value_function_for_action(i, j, action)
        # Return the action with the maximum value function
        return np.argmax(value_function_for_action)


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
    # Create a policy array of the same size as the array, fill with random real numbers between -2pi and 2pi
    policy = np.random.uniform(-2 * np.pi, 2 * np.pi, grid_world.shape)
    plot_2d_array_with_arrows(grid_world, policy)
    if model_type == "Deterministic":
        probability = np.ones(grid_world.shape)
    else:
        probability = np.full(grid_world.shape, .8)
    plot_2d_array_with_grid(grid_world)


# Calling the main function
if __name__ == "__main__":
    if len(sys.argv) <= 1 or (sys.argv[1] != "Deterministic" and sys.argv[1] != "Stochastic"):
        print("Please enter either \"Deterministic\" or \"Stochastic\"")
    else:
        main(sys.argv[1])