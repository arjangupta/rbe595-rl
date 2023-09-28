# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning, Week 5
# Programming Exercise 3: Monte Carlo method

import numpy as np
import matplotlib.pyplot as plt

class MonteCarloES:
    """Monte Carlo Exploring Starts algorithm for estimating optimal policy,
    as given on page 99 of the textbook"""
    def __init__(self, episode_length=1000, num_episodes=1000, gamma=0.95):
        # Randomly initialize policy for back and forward actions (0 and 1)
        self.policy = np.random.randint(2, size=episode_length)
        # Initialize Q(s,a) arbitrarily to real numbers
        self.Q = np.random.rand(episode_length)
        # Initialize returns to empty list
        self.returns = []
        self.num_episodes = num_episodes
        self.gamma = gamma
    
    def run(self):
        """Runs the Monte Carlo algorithm for the specified number of episodes"""
        for i in range(self.num_episodes):
            # Choose random starting state and action
            start_state = np.random.randint(2)
            start_action = self.policy[state]
            # Generate an episode using the current policy
            episode = self.generate_episode(start_state, start_action)
            G = 0
            # For each step in the episode
            for step in reversed(episode):
                # Update G
                G = self.gamma * G + step[2]
                # If the state-action pair is not in the episode
                if (step[0], step[1]) not in self.returns:
                    # Add it to the episode
                    self.returns.append((step[0], step[1]))
                    # Update Q(s,a)
                    self.Q[step[0], step[1]] = G
                    # Update Q(s,a) for each state-action pair in the episode
                    self.update_Q(episode)
                    # Update the policy to be greedy with respect to the new Q(s,a)
                    self.update_policy()


if __name__ == "__main__":
    mc_es = MonteCarloES()
    mc_es.run()
    print(mc_es.Q)
    print(mc_es.policy)
