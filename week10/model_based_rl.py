"""
Authors: Taylor Bergeron and Arjan Gupta
RBE 595 Reinforcement Learning
Programming Exercise 5: Model-Based RL
"""

import matplotlib.pyplot as plt
import numpy as np

class World():

    def __init__(self, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95):
        # Create a grid with input shape
        self.gridworld = np.zeros((height, width))

        # Set the obstacles
        self.gridworld[1:4, 2] = 1
        self.gridworld[4, 5] = 1
        self.gridworld[0:3, 7] = 1

        # encoded in action is: next state, reward
        self.model = np.zeros((height, width, actions, 2))

        #in each action space put tuple with next state and reward
        #actions 0=up 1=right 2=down 3=left
        #populate goal reward with 1
        self.model[0,width-1,0,1]=1
        self.model[0,width-1,1,1]=1
        self.model[0,width-1,3,1]=1
        self.model[0,width-2,1,1]=1 #agent can never be here because obstacle
        self.model[1,width-1,0,1]=1

    def plot_gridworld(self):
        _, ax = plt.subplots()

        # Plot the grid
        ax.imshow(self.gridworld, cmap='binary')

        # Form the grid lines such that they are in the middle of each cell
        ax.set_xticks(np.arange(-.5, self.gridworld.shape[1], 1))
        ax.set_yticks(np.arange(-.5, self.gridworld.shape[0], 1))
        ax.grid(which='both', color='black', linewidth=2)

        plt.show()

    def take_action(self, state, action):
        return self.gridworld[state][action]

class Model():

    def __init__(self, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95):

        # encoded in action is: next state, reward
        self.model = np.zeros((height, width, actions, 2))

        #in each action space put tuple with next state and reward
        #actions 0=up 1=right 2=down 3=left
        #populate goal reward with 1
        self.model[0,width-1,0,1]=1
        self.model[0,width-1,1,1]=1
        self.model[0,width-1,3,1]=1
        self.model[0,width-2,1,1]=1 #agent can never be here because obstacle
        self.model[1,width-1,0,1]=1

    def take_action(self, state, action):
        r = state[0]
        c = state[1]
        return self.model[r,c,action]


if __name__ == "__main__":

    world = World()
    model = Model()
    world.plot_gridworld()
