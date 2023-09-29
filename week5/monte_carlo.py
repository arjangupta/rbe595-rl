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
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

class EpisodeGenerator:
    """Generates episodes for the Monte Carlo algorithms,
    as given in Example 2.2 of the textbook"""
    def __init__(self, num_states=6, num_actions=2, stochastic=True):
        self.max_episode_length = 1000
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
        
    def generate(self, policy, exploring_start=True):
        """Generates an episode for the Monte Carlo algorithm"""
        episode = []

        if exploring_start:
            # Initialize the current state and action as a random start state
            current_state = np.random.randint(self.num_states)
        else:
            # Initialize the current state to 0, so we begin at start of environment interaction
            current_state = 0

        # For each step in the episode
        for _ in range(self.max_episode_length):
            # Get action from policy
            current_action = policy[current_state]

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
            # Get the reward
            if next_state == 0 and current_state != 0:
                reward = 1
            elif next_state == self.num_states - 1 and current_state != self.num_states - 1:
                reward = 5
            else:
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
    def __init__(self, num_episodes=500, gamma=0.7, stochastic=True):
        self.num_states = 6
        self.num_actions = 2

        # Arbitrarily assign policy(s) in A(s), for all s in S
        self.policy = np.random.randint(self.num_actions, size=self.num_states)

        # Initialize Q(s,a) arbitrarily to real numbers, for all s in S, a in A(s)
        self.Q = np.random.rand(self.num_states, self.num_actions)

        # Initialize a Q over time array
        self.Q_arr = np.zeros((num_episodes, self.num_states, self.num_actions))

        # Initialize returns to shape of Q, with empty lists
        self.returns = np.empty_like(self.Q, dtype=list)
        for i in range(self.returns.shape[0]):
            for j in range(self.returns.shape[1]):
                self.returns[i, j] = []

        # Initialize number of episodes, and discount factor
        self.num_episodes = num_episodes
        self.gamma = gamma

        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(self.num_states, self.num_actions, stochastic)

        # Set verbose flag to False
        self.show_pi_q = False

    def set_show_pi_q(self, show):
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
        for e in trange(self.num_episodes):
            # Generate an episode using the current policy
            episode = self.episode_generator.generate(self.policy)
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
            # Add the Q values to the Q over time array
            self.Q_arr[e, :, :] = self.Q

        if self.show_pi_q:
            print(f"Finished running Monte Carlo ES algorithm with {self.num_episodes} episodes")
            print("Final policy:")
            print(self.policy)
            print("Final Q values:")
            print(self.Q)


class OnPolicyFirstVisitMC:
    """On-policy first visit monte-carlo for estimating optimal policy,
    as given on page 99 of the textbook"""
    def __init__(self, num_episodes=500, gamma=0.7, epsilon = 0.1, stochastic=True):
        self.num_states = 6
        self.num_actions = 2
        self.epsilon = epsilon

        # Randomly initialize policy for back and forward actions (0 and 1)
        self.policy = np.random.randint(self.num_actions, size=self.num_states)

        # Initialize Q(s,a) arbitrarily to real numbers
        self.Q = np.random.rand(self.num_states, self.num_actions)

        # Initialize a Q over time array
        self.Q_arr = np.zeros((num_episodes, self.num_states, self.num_actions))

        # Initialize returns to shape of Q, with empty lists
        self.returns = np.empty_like(self.Q, dtype=list)
        for i in range(self.returns.shape[0]):
            for j in range(self.returns.shape[1]):
                self.returns[i, j] = []

        # Initialize number of episodes, and discount factor
        self.num_episodes = num_episodes
        self.gamma = gamma

        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(self.num_states, self.num_actions, stochastic)

        # Set verbose flag to False
        self.show_pi_q = False

    def set_show_pi_q(self, show):
        """Sets flag to show the policy and Q values"""
        self.show_pi_q = show

    def run(self):
        """Runs the Monte Carlo algorithm for the specified number of episodes"""
        if self.show_pi_q:
            print("Initial policy:")
            print(self.policy)
            print("Initial Q values:")
            print(self.Q)
            print(f"Running On-policy First-visit MC Control with {self.num_episodes} episodes...")
        for e in trange(self.num_episodes):
            # Generate an episode using the current policy
            episode = self.episode_generator.generate(self.policy, False)
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
                    # Epsilon-greedy policy improvement
                    A_star = np.argmax(self.Q[state, :])
                    if action == A_star:
                        self.policy[state] = 1 - self.epsilon + (self.epsilon/self.num_actions)
                    # taking non-greedy action
                    else:
                        self.policy[state] = self.epsilon/self.num_actions

            # Add the Q values to the Q over time array
            self.Q_arr[e, :, :] = self.Q

        if self.show_pi_q:
            print(f"Finished running On-policy First-visit MC Control algorithm with {self.num_episodes} episodes")
            print("Final policy:")
            print(self.policy)
            print("Final Q values:")
            print(self.Q)

def plot_Qs(Q_arr, max_episodes, algo_name):
    """For each of the 6 states do the following:
    1. Iterate through Q_arr for that state
    2. Sub-plot the Q value for both actions over the number of episodes
    3. Show the episodes in multiples of 5"""
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    # Set title for entire figure
    fig.suptitle(f"{algo_name}: Q Values over {max_episodes} Episodes")
    for i in range(6):
        row = i // 2
        col = i % 2
        # Plot actions with labels
        axs[row, col].plot(Q_arr[:, i, 0], label="Back")
        axs[row, col].plot(Q_arr[:, i, 1], label="Forward")
        # Set subplot title
        axs[row, col].set_title(f"State {i}")
        # Show legend
        axs[row, col].legend()
    plt.show()

def main(algorithm):
    # Run Monte Carlo ES for various numbers of episodes
    Q_arr = []
    max_episodes = 100
    if algorithm == "1":
        # Run Monte Carlo ES for various numbers of episodes
        mc_es = MonteCarloES()
        mc_es.set_show_pi_q(True)
        mc_es.run()
        # Plot the Q values over number of episodes
        plot_Qs(mc_es.Q_arr, mc_es.num_episodes, "Monte Carlo ES")
    else:
        # Run On-policy First-visit MC Control for various numbers of episodes
        op_fv_mc = OnPolicyFirstVisitMC(epsilon=0.1)
        op_fv_mc.set_show_pi_q(True)
        op_fv_mc.run()
        # Plot the Q values over number of episodes
        plot_Qs(op_fv_mc.Q_arr, op_fv_mc.num_episodes, "On-policy First-visit MC Control")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please specify the algorithm by entering the associated number as a run argument: 1) Monte Carlo ES 2) On-policy First-visit MC Control")
    else:
        main(sys.argv[1])
