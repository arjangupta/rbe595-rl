# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning, Week 5
# Programming Exercise 3: Monte Carlo method

# We are simulating the robot in Example 2.2 of the textbook, which is a robot
# that can move forward or backward in a one-dimensional, 5 state world.
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
    def __init__(self, episode_length=1000):
        self.episode_length = episode_length
        self.num_states = 5
        self.num_actions = 2
        self.expected_dir_prob = 0.8
        self.opposite_dir_prob = 0.05
        self.stay_prob = 0.15
        
    def generate(self):
        """Generates an episode for the Monte Carlo algorithm"""
        episode = []
        # Choose random starting state and action
        start_state = np.random.randint(1, self.num_states)
        start_action = np.random.randint(self.num_actions)
        # Initialize the current state and action
        current_state = start_state
        current_action = start_action
        # For each step in the episode
        for i in range(self.episode_length):
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
            # If the robot is in state 5
            elif next_state == 5:
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
        # If the robot is in state 5
        elif state == 5:
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
        # Randomly initialize policy for back and forward actions (0 and 1)
        self.policy = np.random.randint(2, size=episode_length)
        # Initialize Q(s,a) arbitrarily to real numbers
        self.Q = np.random.rand(5, 2)
        # Initialize returns to empty list
        self.returns = []
        # Initialize episode length, number of episodes, and discount factor
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.gamma = gamma
        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(self.episode_length)
    
    def run(self):
        """Runs the Monte Carlo algorithm for the specified number of episodes"""
        for i in range(self.num_episodes):
            # Generate an episode using the current policy
            episode = self.episode_generator.generate()
            G = 0
            # For each step in the episode
            for step in reversed(episode):
                # Update G
                G = self.gamma * G + step[2]
                # If the state-action pair is not in the episode
                if (step[0], step[1], _) not in episode:
                    # Add it to the returns
                    self.returns.append((step[0], step[1]))
                    # Update Q(s,a) using the new return
                    self.Q[step[0], step[1]] = G
                    # Update policy
                    self.policy[step[0]] = np.argmax(self.Q)


if __name__ == "__main__":
    mc_es = MonteCarloES()
    mc_es.run()
    print(mc_es.Q)
    print(mc_es.policy)
