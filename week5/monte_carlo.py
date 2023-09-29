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
from tqdm import trange

class EpisodeGenerator:
    """Generates episodes for the Monte Carlo algorithms,
    as given in Example 2.2 of the textbook"""
    def __init__(self, policy, num_states=6, num_actions=2, stochastic=True):
        self.max_episode_length = 100000
        self.num_states = num_states
        self.num_actions = num_actions
        self.stochastic = stochastic
        if self.stochastic:
            self.expected_dir_prob = 0.8
            self.opposite_dir_prob = 0.05
            self.stay_prob = 0.15
        else:
            self.expected_dir_prob = 1
            self.opposite_dir_prob = 0
            self.stay_prob = 0
        self.policy = policy
        
    def generate(self):
        """Generates an episode for the Monte Carlo algorithm"""
        episode = []

        # Initialize the current state and action as a random start state
        current_state = np.random.randint(self.num_states)

        # For each step in the episode
        for _ in range(self.max_episode_length):

            # Get action from policy
            current_action = self.policy[current_state]

            # Generate a random number for the direction the robot moves
            random_number = np.random.rand()

            if random_number < self.expected_dir_prob:
                # The robot moves in the direction it chooses
                next_state = self.transition(current_state, current_action)
            elif random_number < self.expected_dir_prob + self.opposite_dir_prob:
                # The robot moves in the opposite direction
                current_action *= -1
                next_state = self.transition(current_state, current_action)
            else:
                # The robot stays in the same place
                next_state = current_state

            if next_state == 0 and current_state != 0:
                # The robot receives a reward of 1
                reward = 1
            elif next_state == self.num_states - 1 and current_state != self.num_states - 1:
                # The robot receives a reward of 5
                reward = 5
            else:
                # The robot receives a reward of 0
                reward = 0

            # Add the step to the episode
            episode.append((current_state, current_action, reward))

            # Update the current state and action
            current_state = next_state

            # If we have reached one of the terminal states, stop generating the episode
            if current_state == 0 or current_state == self.num_states - 1:
                break

        return episode
    
    def transition(self, state, action):
        """Returns the next state given the current state and action"""
        if state == 0 and action == 0:
            return 0
        elif state == self.num_states - 1 and action == 1:
            return self.num_states - 1
        else:
            return state + action


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

        # Initialize number of episodes, and discount factor
        self.num_episodes = num_episodes
        self.gamma = gamma

        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(self.policy, self.num_states, self.num_actions)

    def show_pi_q(self, show=True):
        """Sets flag to show the policy and Q values"""
        self.show_pi_q = show


    def run(self):
        """Runs the Monte Carlo algorithm for the specified number of episodes"""
        if self.show_pi_q:
            print("Initial policy:")
            print(self.policy)
            print("Initial Q values:")
            print(self.Q)
        print(f"Running Monte Carlo ES algorithm with {self.num_episodes} episodes...")
        for i in trange(self.num_episodes):
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
        print("Finished running Monte Carlo ES algorithm")
        if self.show_pi_q:
            print("Final policy:")
            print(self.policy)
            print("Final Q values:")
            print(self.Q)

if __name__ == "__main__":
    mc_es = MonteCarloES()
    mc_es.show_pi_q(True)
    mc_es.run()
