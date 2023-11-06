"""
Authors: Taylor Bergeron and Arjan Gupta
RBE 595 Reinforcement Learning
Programming Exercise 5: Model-Based RL
"""
import random

import matplotlib.pyplot as plt
import numpy as np

class World():

    def __init__(self, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95):
        # Create a grid with input shape
        self.rows = height
        self.cols = width
        self.gridworld = np.zeros((self.rows, self.cols))
        self.goal = [0, self.cols-1]
        self.start = [2, 0]

        # Set the obstacles
        self.gridworld[1:4, 2] = 1
        self.gridworld[4, 5] = 1
        self.gridworld[0:3, 7] = 1

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
        # actions 0=up 1=right 2=down 3=left
        next_state = state #[0,0]
        if action==0:
            if not state[0]==0 and not self.gridworld[state[0]-1, state[1]]==1:
                next_state = [state[0]-1, state[1]]
        if action==1:
            if not state[1]==self.cols-1 and not self.gridworld[state[0], state[1]+1]==1:
                next_state = [state[0], state[1]+1]
        if action==2:
            if not state[0]==self.rows-1 and not self.gridworld[state[0]+1, state[1]]==1:
                next_state = [state[0]+1, state[1]]
        if action==3:
            if not state[1]==0 and not self.gridworld[state[0], state[1]-1]==1:
                next_state = [state[0], state[1]-1]

        return next_state, int(next_state == self.goal)


class Model():

    def __init__(self, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95):

        # encoded in action is: next state, reward
        self.model = np.zeros((height, width, actions, 2))
        # fill model with -1s so we can encode visited, nextstate=-1 and reward=-1 is unvisited
        self.model.fill(-1)

        #in each action space put tuple with next state and reward
        #actions 0=up 1=right 2=down 3=left

        #REMOVE BELOW: model needs to figure this out
        # in each action space put tuple with next state and reward
        # actions 0=up 1=right 2=down 3=left
        # populate goal reward with 1
        # self.model[0, width - 1, 0, 1] = 1
        # self.model[0, width - 1, 1, 1] = 1
        # self.model[0, width - 1, 3, 1] = 1
        # self.model[0, width - 2, 1, 1] = 1  # agent can never be here because obstacle
        # self.model[1, width - 1, 0, 1] = 1

    def take_action(self, state, action):
        r = state[0]
        c = state[1]
        return self.model[r,c,action]

    def get_random_previously_observed_state_and_action(self):
        #TODO: randomly select state where next state and reward is not -1
        # self.model[:,:,:] != -1
        return 0,0

class TabularDynaQ():
    def __init__(self, model, world, episodes = 50, planning_steps = 5, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.model = model
        self.world = world
        self.episodes = episodes
        self.planning_steps = planning_steps
        self.Q = np.zeros((height, width, actions))
        self.epsilon = epsilon
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma

    def run(self):
        for ep in range(self.episodes):
            goal = False
            state = self.world.start
            while not goal:
                # action = epsilon-greedy(S,Q)
                dice_roll = random.uniform(0, 1)
                if dice_roll <= self.epsilon:
                    action = random.randint(0, self.actions-1)
                else:
                    #TODO: make sure below works as intended
                    action = np.argmax(self.Q[state[0], state[1], :])
                next_state, reward = world.take_action(state, action)
                max_a_Q =np.argmax(self.Q[next_state[0], next_state[1], :])
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], max_a_Q] - self.Q[state[0], state[1], action])
                #update model
                state = next_state
                for planning_step in range(self.planning_steps):
                    s,a = self.model.get_random_previously_observed_state_and_action()

if __name__ == "__main__":

    world = World()
    model = Model()
    # world.plot_gridworld()
    dq = TabularDynaQ(model=model, world=world)

