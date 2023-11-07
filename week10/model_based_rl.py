"""
Authors: Taylor Bergeron and Arjan Gupta
RBE 595 Reinforcement Learning
Programming Exercise 5: Model-Based RL
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import random
import time

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
    
    def update_gridworld(self, Q):
        """Update the gridworld with an arrow showing the best action at each state"""
        # Create a grid with input shape
        self.rows = Q.shape[0]
        self.cols = Q.shape[1]
        self.gridworld = np.zeros((self.rows, self.cols))

        # Set the obstacles
        self.gridworld[1:4, 2] = 1
        self.gridworld[4, 5] = 1
        self.gridworld[0:3, 7] = 1

        # Set the goal
        self.gridworld[0, self.cols-1] = 2

    def plot_gridworld(self, Q):
        _, ax = plt.subplots()
        verbose = False

        # Iterate through the grid and set arrow directions
        for r in range(self.rows):
            for c in range(self.cols):
                if r==5 and c == 5:
                    if verbose:
                        print(f"Q for state 5,5, {Q[r, c, :]}")
                if np.argmax(Q[r, c, :]) == 0:
                    ax.annotate('↑', xy=(c, r), horizontalalignment='center', verticalalignment='center')
                elif np.argmax(Q[r, c, :]) == 1:
                    ax.annotate('→', xy=(c, r), horizontalalignment='center', verticalalignment='center')
                elif np.argmax(Q[r, c, :]) == 2:
                    ax.annotate('↓', xy=(c, r), horizontalalignment='center', verticalalignment='center')
                elif np.argmax(Q[r, c, :]) == 3:
                    ax.annotate('←', xy=(c, r), horizontalalignment='center', verticalalignment='center')

        # Plot the grid
        ax.imshow(self.gridworld, cmap='binary')

        # Form the grid lines such that they are in the middle of each cell
        ax.set_xticks(np.arange(-.5, self.gridworld.shape[1], 1))
        ax.set_yticks(np.arange(-.5, self.gridworld.shape[0], 1))
        ax.grid(which='both', color='black', linewidth=2)

        plt.show()

    def take_action(self, state, action):
        # Actions are: 0=up 1=right 2=down 3=left
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
        # Encoded in action is: next state[0], next state[1], reward
        self.model = np.zeros((height, width, actions, 3))
        # Fill model with -1s so we can encode visited, nextstate=-1 and reward=-1 is unvisited
        self.model.fill(-1)

    def take_action(self, state, action):
        r = state[0]
        c = state[1]
        # returns return and next state
        return self.model[r,c,action]

    def get_random_previously_observed_state_and_action(self):
        # Randomly select state where next state and reward is not -1
        matches = np.where(self.model != -1)
        if matches is not None and len(matches)>=3:
            locations = list()
            for i in range(len(matches[0])):
                # Does this to get rid of duplicates from reward and next state being encoded under actions
                if i%3 == 0:
                    locations.append([matches[0][i], matches[1][i], matches[2][i]])
            random_index = random.randint(0, len(locations)-1) #randint is inclusive
            return locations[random_index]
        print("no observed states in model")
        return [0,0,0]

    def set_next_state_and_reward(self, state, action, next_state, reward):
        self.model[state[0], state[1], action, 0] = next_state[0]
        self.model[state[0], state[1], action, 1] = next_state[1]
        self.model[state[0], state[1], action, 2] = reward

class TabularDynaQ():
    def __init__(self, model, world, episodes = 50, planning_steps = 50, height=6, width=9, actions=4, alpha=0.1, epsilon=0.1, gamma=0.95, verbose=False):
        self.model = model
        self.world = world
        self.episodes = episodes
        self.planning_steps = planning_steps
        self.Q = np.zeros((height, width, actions))
        self.epsilon = epsilon
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.steps_per_episode = list()
        self.verbose = verbose

    def run(self):
        print("Running Dyna-Q for {} episodes with {} planning steps".format(self.episodes, self.planning_steps))
        for _ in trange(self.episodes):
            state = self.world.start
            goal = False
            steps = 0
            # Take steps until goal is reached
            while not goal:
                dice_roll = random.uniform(0, 1)
                if dice_roll <= self.epsilon:
                    action = random.randint(0, self.actions-1) # randint is inclusive
                else:
                    all_actions_for_state = self.Q[state[0], state[1], :]
                    if self.verbose:
                        print(all_actions_for_state)
                    if all(item == all_actions_for_state[0] for item in all_actions_for_state):
                        if self.verbose:
                            print("no q max - all actions have same value. choosing randomly")
                        # If all the q values for the actions are the same, pick an action at random
                        action = random.randint(0, self.actions - 1)  # randint is inclusive
                    else:
                        action = np.argmax(self.Q[state[0], state[1], :])
                if self.verbose:
                    print("chosen action: %s" % action)
                next_state, reward = self.world.take_action(state, action)
                max_a_Q =np.argmax(self.Q[next_state[0], next_state[1], :])
                # Update Q
                self.Q[state[0], state[1], action] += self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], max_a_Q] - self.Q[state[0], state[1], action])
                # Update model
                self.model.set_next_state_and_reward(state, action, next_state, reward)
                if self.verbose:
                    print("state: %s" % state)
                # Check if goal is reached
                if state == self.world.goal:
                    if self.verbose:
                        print("goal reached, reward %i" % reward)
                    goal = True
                state = next_state
                if self.verbose:
                    print("next state: %s" % state)

                # Planning phase
                for planning_step in range(self.planning_steps):
                    if self.verbose:
                        print("planning step: {}".format(planning_step))
                    state_and_action = self.model.get_random_previously_observed_state_and_action()
                    s = [state_and_action[0], state_and_action[1]]
                    a = state_and_action[2]
                    if self.verbose:
                        print("current planning state: %s and action %s" % (s, a))
                    next_state_and_reward = self.model.model[s[0], s[1], a]
                    next_state = [int(next_state_and_reward[0]), int(next_state_and_reward[1])]
                    reward = next_state_and_reward[2]
                    max_a_Q = np.argmax(self.Q[next_state[0], next_state[1], :])
                    if self.verbose:
                        print("Q before update: %s" % self.Q[s[0], s[1], a])
                    self.Q[s[0], s[1], a] += self.alpha * (
                                reward + self.gamma * self.Q[next_state[0], next_state[1], max_a_Q] - self.Q[
                            s[0], s[1], a])
                    if self.verbose:
                        print("Q after update: %s" % self.Q[s[0], s[1], a])
                # Increment steps
                steps+=1
            self.steps_per_episode.append(steps)
        if self.verbose:
            print("Q: {}".format(self.Q))
            print(self.steps_per_episode)
        world.update_gridworld(self.Q)

def plot_steps_per_episode(dq0, dq5, dq50):
    plt.plot(dq0, label='0 planning steps (direct RL only)')
    plt.plot(dq5, label='5 planning steps')
    plt.plot(dq50, label='50 planning steps')
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.show()

if __name__ == "__main__":

    world = World()
    model = Model()

    show_gridworld = False

    dq0 = TabularDynaQ(model=model, world=world, episodes = 50, planning_steps = 0)
    dq0.run()
    if show_gridworld:
        # Show the gridworld with arrows
        world.plot_gridworld(dq0.Q)

    dq5 = TabularDynaQ(model=model, world=world, episodes=50, planning_steps=5)
    dq5.run()
    if show_gridworld:
        # Show the gridworld with arrows
        world.plot_gridworld(dq5.Q)

    dq50 = TabularDynaQ(model=model, world=world, episodes=50, planning_steps=50)
    dq50.run()
    if show_gridworld:
        # Show the gridworld with arrows
        world.plot_gridworld(dq50.Q)

    plot_steps_per_episode(dq0.steps_per_episode, dq5.steps_per_episode, dq50.steps_per_episode)




