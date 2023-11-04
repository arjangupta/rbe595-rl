"""
Authors: Taylor Bergeron and Arjan Gupta
RBE 595 Reinforcement Learning
Programming Exercise 5: Model-Based RL
"""

import matplotlib.pyplot as plt
import numpy as np

class World():

    def __init__(self):
        # Create a 6x9 grid
        self.gridworld = np.zeros((6, 9))

        # Set the obstacles
        self.gridworld[1:4, 2] = 1
        self.gridworld[4, 5] = 1
        self.gridworld[0:3, 7] = 1

        # plot_gridworld(gridworld)

    def plot_gridworld(self):
        _, ax = plt.subplots()

        # Plot the grid
        ax.imshow(self.gridworld, cmap='binary')

        # Form the grid lines such that they are in the middle of each cell
        ax.set_xticks(np.arange(-.5, self.gridworld.shape[1], 1))
        ax.set_yticks(np.arange(-.5, self.gridworld.shape[0], 1))
        ax.grid(which='both', color='black', linewidth=2)

        plt.show()

if __name__ == "__main__":
    world = World()
    world.plot_gridworld()
