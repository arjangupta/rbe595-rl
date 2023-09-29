# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning, Week 5
# Programming Exercise 3: Monte Carlo method

# We are simulating the robot in Example 2.2 of the textbook, which is a robot
# that can move forward or backward in a one-dimensional, 6 state world.
# The transition dynamics are stochastic, such that 0.8 of the time the robot
# moves in the direction it chooses, and 0.05 of the time it moves in the opposite
# direction, and 0.15 of the time it stays in the same place. The robot receives
# a reward of 5 in state 5, where there is a can. The robot receives a reward of
# 1 if it reaches state 0, where there is a charger for its battery. The robot
# receives a reward of 0 everywhere else.

import numpy as np
import matplotlib.pyplot as plt

class EpisodeGenerator:
    """Generates episodes for the Monte Carlo algorithms,
    as given in Example 2.2 of the textbook"""
    def __init__(self, episode_length=1000, num_states=6, num_actions=2):
        self.episode_length = episode_length
        self.num_states = num_states
        self.num_actions = num_actions
        self.expected_dir_prob = 0.8
        self.opposite_dir_prob = 0.05
        self.stay_prob = 0.15
        
    def generate(self):
        """Generates an episode for the Monte Carlo algorithm"""
        episode = []
        # Choose random starting state and action
        start_state = np.random.randint(self.num_states)
        start_action = np.random.randint(self.num_actions)
        # Initialize the current state and action
        current_state = start_state
        current_action = start_action
        # For each step in the episode
        for _ in range(self.episode_length):
            # Generate a random number
            random_number = np.random.rand()
            # If the random number is less than expected direction probability
            if random_number < self.expected_dir_prob:
                # The robot moves in the direction it chooses
                next_state = current_state + current_action
            # If the random number is less than expected + opposite direction probability
            elif random_number < self.expected_dir_prob + self.opposite_dir_prob:
                # The robot moves in the opposite direction
                next_state = current_state - current_action
            # If the random number is less than 1
            else:
                # The robot stays in the same place
                next_state = current_state
            # If the robot is in state 0
            if next_state == 0:
                # The robot receives a reward of 1
                reward = 1
            # If the robot is in rightmost state
            elif next_state == self.num_states - 1:
                # The robot receives a reward of 5
                reward = 5
            # If the robot is in any other state
            else:
                # The robot receives a reward of 0
                reward = 0
            # Add the step to the episode
            episode.append((current_state, current_action, reward))
            # Update the current state and action
            current_state = next_state
            current_action = self.choose_action(current_state)
        return episode
    
    def choose_action(self, state):
        """Chooses an action for the robot to take in the given state"""
        # If the robot is in state 0
        if state == 0:
            # The robot can only move forward
            return 1
        # If the robot is in rightmost-state
        elif state == self.num_states - 1:
            # The robot can only move backward
            return 0
        # If the robot is in any other state
        else:
            # The robot can move forward or backward
            return np.random.randint(2)

class MonteCarloES:
    """Monte Carlo Exploring Starts algorithm for estimating optimal policy,
    as given on page 99 of the textbook"""
    def __init__(self, episode_length=1000, num_episodes=1000, gamma=0.95):
        self.num_states = 6
        self.num_actions = 2
        # Randomly initialize policy for back and forward actions (0 and 1)
        self.policy = np.random.randint(self.num_actions, size=self.num_states)
        # Initialize Q(s,a) arbitrarily to real numbers
        self.Q = np.random.rand(self.num_states, self.num_actions)
        # Initialize returns to shape of Q, with empty lists
        self.returns = np.empty_like(self.Q, dtype=list)
        for i in range(self.returns.shape[0]):
            for j in range(self.returns.shape[1]):
                self.returns[i, j] = []
        # Initialize episode length, number of episodes, and discount factor
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.gamma = gamma
        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(self.episode_length, self.num_states, self.num_actions)
    
    def run(self):
        """Runs the Monte Carlo algorithm for the specified number of episodes"""
        for i in range(self.num_episodes):
            # Generate an episode using the current policy
            episode = self.episode_generator.generate()
            G = 0
            # For each step in the episode
            for i, step in enumerate(reversed(episode)):
                # Get the state, action, and reward
                state, action, reward = step
                # Update the return
                G = self.gamma * G + reward
                # Iterate through all previous steps in the episode, and
                # check if the pair (state, action) appears in any previous step
                pair_occurred = False
                for j in range(len(episode) - i - 1):
                    if episode[j][0] == state and episode[j][1] == action:
                        pair_occurred = True
                # "Unless the pair (state, action) appears in any previous step"
                if not pair_occurred:
                    # Add the return to the returns
                    self.returns[state, action].append(G)
                    # Update the Q value
                    self.Q[state, action] = np.mean(self.returns[state, action])
                    # Update the policy
                    self.policy[state] = np.argmax(self.Q[state, :])
                


if __name__ == "__main__":
    mc_es = MonteCarloES()
    print(mc_es.Q)
    print(mc_es.policy)
    mc_es.run()
    print(mc_es.Q)
    print(mc_es.policy)
