# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning, Week 7
# Programming Assignment for Temporary-Difference Learning

# In this assignment, we will implement both the SARSA and Q-learning algorithms for the
# cliff-walking problem described in Example 6.6 of the textbook (page 132).

# The cliff-walking problem is a gridworld with a 4x12 grid of states. The agent starts in the
# bottom left corner of the grid and must navigate to the bottom right corner. The agent can
# move in any of the four cardinal directions, but if it moves into the cliff (the region of
# states in the bottom row, excluding the bottom left and bottom right corners), it will fall
# off and be sent back to the start. The agent receives a reward of -1 for each step it takes
# that does not result in falling off the cliff. The agent receives a reward of -100 for falling
# off the cliff. The agent receives a reward of 0 for reaching the goal state.

import numpy as np
import matplotlib.pyplot as plt

def plot_gridworld(path1, path2):
    """
    Plots a gridworld for the cliff-walking problem.

    Args:
        gridworld (list): A 4x12 grid of states, represented as a list of lists.
    """
    plt.figure()
    plt.gcf().set_size_inches(12, 4)
    plt.title("Cliff-Walking Gridworld")
    plt.xlim(0, 12)
    plt.ylim(0, 4)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(0, 4, 1))
    plt.grid(True)
    for j in range(12):
        if j != 0 and j != 11:
            plt.gca().add_patch(plt.Rectangle((j, 0), 1, 1, facecolor="grey"))
    # Plot the paths and show the legend
    plt.plot([x for x, _ in path1], [y for _, y in path1], "b-", label="Path 1")
    plt.plot([x for x, _ in path2], [y for _, y in path2], "r-", label="Path 2")
    plt.legend()
    # Put a big S at the start state
    plt.text(0.5, 0.5, "S", ha="center", va="center", fontsize=20)
    # Put a big G at the goal state
    plt.text(11.5, 0.5, "G", ha="center", va="center", fontsize=20)
    plt.show()

def main():
    print("TD Programming Assignment")

    # Generate 2 example paths, starting at (0.5, 0.5) and ending at (11.5, 0.5), make them take different routes
    path1 = [[0.5,0.5], [1.5,2.5], [2.5,2.5], [2.5,2.5], [4.5,2.5], [5.5,2.5], [6.5,2.5], [7.5,2.5], [8.5,2.5], [9.5,2.5], [10.5,2.5], [11.5,0.5]]
    path2 = [[0.5,0.5], [0.5,1.5], [2.5,2.5], [3.5,3.5], [4.5,3.5], [5.5,3.5], [6.5,3.5], [7.5,3.5], [8.5,3.5], [9.5,3.5], [10.5,3.5], [11.5,0.5]]

    plot_gridworld(path1, path2)

if __name__ == "__main__":
    main()
