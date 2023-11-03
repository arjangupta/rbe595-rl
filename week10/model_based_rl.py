"""
Authors: Taylor Bergeron and Arjan Gupta
RBE 595 Reinforcement Learning
Programming Exercise 5: Model-Based RL
"""

import matplotlib.pyplot as plt
import numpy as np

def create_gridworld():
    # Create a 6x9 grid
    gridworld = np.zeros((6, 9))

    # Set the obstacles
    gridworld[1:4, 2] = 1
    gridworld[4,5] = 1
    gridworld[0:3, 7] = 1

    fig, ax = plt.subplots()

    # Plot the grid
    ax.imshow(gridworld, cmap='binary')

    # Form the grid lines such that they are in the middle of each cell
    ax.set_xticks(np.arange(-.5, gridworld.shape[1], 1))
    ax.set_yticks(np.arange(-.5, gridworld.shape[0], 1))
    ax.grid(which='both', color='black', linewidth=2)

    plt.show()

if __name__ == "__main__":
    create_gridworld()
