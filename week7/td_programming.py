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

def generate_gridworld():
    """
    Generates a gridworld for the cliff-walking problem.

    Returns:
        gridworld (list): A 4x12 grid of states, represented as a list of lists.
    """
    gridworld = []
    for i in range(4):
        row = []
        for j in range(12):
            row.append((i, j))
        gridworld.append(row)
    return gridworld

def plot_gridworld(gridworld, path1, path2):
    """
    Plots a gridworld for the cliff-walking problem.

    Args:
        gridworld (list): A 4x12 grid of states, represented as a list of lists.
    """
    plt.figure()
    plt.title("Cliff-Walking Gridworld")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xlim(0, 12)
    plt.ylim(0, 4)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(0, 4, 1))
    plt.grid(True)
    for j in range(12):
        if j != 0 and j != 11:
            plt.gca().add_patch(plt.Rectangle((j, 0), 1, 1, facecolor="grey"))
    # Plot the path
    for i in range(len(path1) - 1):
        plt.plot([path1[i][1] + 0.5, path1[i + 1][1] + 0.5], [path1[i][0] + 0.5, path1[i + 1][0] + 0.5], color="blue")
    for i in range(len(path2) - 1):
        plt.plot([path2[i][1] + 0.5, path2[i + 1][1] + 0.5], [path2[i][0] + 0.5, path2[i + 1][0] + 0.5], color="red")
    # Show legend
    plt.plot([], [], color="blue", label="SARSA")
    plt.plot([], [], color="red", label="Q-learning")
    # Put a big S at the start state
    plt.text(0.5, 0.5, "S", ha="center", va="center", fontsize=20)
    # Put a big G at the goal state
    plt.text(11.5, 0.5, "G", ha="center", va="center", fontsize=20)
    plt.show()

def main():
    print("TD Programming Assignment")

    # Generate 2 example paths, taking different routes to the goal
    path1 = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (3, 6),
             (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)]
    path2 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
             (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)]

    gridworld = generate_gridworld()
    plot_gridworld(gridworld, path1, path2)

if __name__ == "__main__":
    main()
