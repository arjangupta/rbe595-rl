# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning, Week 7
# Programming Assignment for Temporal-Difference Learning

# In this assignment, we will implement both the SARSA and Q-learning algorithms for the
# cliff-walking problem described in Example 6.6 of the textbook (page 132).

# The cliff-walking problem is a gridworld with a 4x12 grid of states. The agent starts in the
# bottom left corner of the grid and must navigate to the bottom right corner. The agent can
# move in any of the four cardinal directions, but if it moves into the cliff (the region of
# states in the bottom row, excluding the bottom left and bottom right corners), it will fall
# off and be sent back to the start. The agent receives a reward of -1 for each step it takes
# that does not result in falling off the cliff. The agent receives a reward of -100 for falling
# off the cliff. The agent receives a reward of 0 for reaching the goal state.
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Constants for the cliff-walking problem
NUM_ACTIONS = 4
X_DIM = 12
Y_DIM = 4

class QLearningAgent:
    """
    A Q-learning agent that learns to navigate the cliff-walking problem.
    """
    def __init__(self, alpha=0.2, epsilon=0.1, gamma=0.95, num_episodes=500, num_runs=50, decay_rate=0.9, use_epsilon_decay=False):
        """Initializes the Q-learning agent.
            alpha (float): The learning rate.
            epsilon (float): The probability of taking a random action.
            gamma (float): The discount factor.
            num_episodes (int): The number of episodes to train for."""
        self.alpha = alpha
        self.start_epsilon = epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_runs = num_runs
        self.Q = np.zeros((X_DIM, Y_DIM, NUM_ACTIONS))
        self.path = []
        self.sum_of_rewards_during_episodes = np.zeros(num_episodes)
        self.use_epsilon_decay = use_epsilon_decay
        if self.use_epsilon_decay and self.alpha > 0.05:
            self.alpha = 0.01
        self.decay_rate = decay_rate

    def choose_action(self, state, learning=True):
        """
        Chooses an action for the agent to take derived from the Q-table.
        """
        if np.random.rand() < self.epsilon and learning:
            # Take a random action
            return np.random.randint(NUM_ACTIONS)
        else:
            # If state contains non integer values, show them and report error
            if not isinstance(state[0], int) or not isinstance(state[1], int):
                print("State: {}".format(state))
                raise ValueError("State should be an integer tuple")
            # Take the action with the highest Q-value
            return np.argmax(self.Q[state])

    def epsilon_decay(self, ep):
        if self.use_epsilon_decay:
            self.epsilon = self.start_epsilon * (self.decay_rate ** ep)

    def take_action(self, state, action):
        """
        Takes an action and returns the new state and reward.
        """
        if action == 0:
            # Move right
            new_state = (state[0] + 1, state[1])
        elif action == 1:
            # Move up
            new_state = (state[0], state[1] + 1)
        elif action == 2:
            # Move left
            new_state = (state[0] - 1, state[1])
        elif action == 3:
            # Move down
            new_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Invalid action: {}".format(action))
        # Check if the new state is out of bounds
        if new_state[0] < 0 or new_state[0] >= X_DIM or new_state[1] < 0 or new_state[1] >= Y_DIM:
            # Stay in the same state and get a reward of -1
            return state, -1
        # Check if the new state is in the cliff
        if new_state[1] == 0 and new_state[0] != 0 and new_state[0] != X_DIM - 1:
            # Go back to the start and get a reward of -100
            return (0, 0), -100
        # Otherwise, return the new state and a reward of -1
        return new_state, -1

    def episode_finished(self, state):
        """
        Checks if the episode is finished - i.e. if the agent has reached the goal state or fallen off the cliff.
        """
        x = state[0]
        y = state[1]
        if x == X_DIM - 1 and y == 0:
            # Reached the goal state
            return True
        if x != 0 and x != X_DIM - 1 and y == 0:
            # Fell off the cliff
            return True
        return False

    def learn(self):
        """Trains the agent using the Q-learning algorithm."""
        print("Training Q-learning agent...")
        for run in trange(self.num_runs):
            self.epsilon = self.start_epsilon
            for episode in range(self.num_episodes):
                S = (0, 0)
                A = self.choose_action(S)
                current_episode_reward_sum = 0
                while not self.episode_finished(S):
                    S_prime, R = self.take_action(S, A)
                    A_prime = self.choose_action(S_prime)
                    self.Q[S][A] += self.alpha * (R + self.gamma * np.max(self.Q[S_prime]) - self.Q[S][A])
                    S = S_prime
                    A = A_prime
                    current_episode_reward_sum += R
                self.sum_of_rewards_during_episodes[episode] += current_episode_reward_sum
                self.epsilon_decay(episode)
        self.sum_of_rewards_during_episodes /= self.num_runs

    def get_path(self, start_state=(0, 0), end_state=(11, 0)):
        """
        Returns a path that the agent takes from the start state to the end state.
        """
        self.path = []
        S = start_state
        self.path.append(S)
        while not self.episode_finished(S):
            A = self.choose_action(S, learning=False)
            S_prime, _ = self.take_action(S, A)
            S = S_prime
            self.path.append(S)
        self.path.append(end_state)
        return self.path

    def get_sum_of_rewards_during_episodes(self):
        """
        Returns the sum of rewards during each episode.
        """
        return self.sum_of_rewards_during_episodes

class SARSAAgent:
    """
    A SARSA agent that learns to navigate the cliff-walking problem.
    """
    def __init__(self, alpha=0.2, epsilon=0.1, gamma=0.95, num_episodes=500, num_runs=50, decay_rate=0.9, use_epsilon_decay=False):
        """Initializes the SARSA agent.
            alpha (float): The learning rate.
            epsilon (float): The probability of taking a random action.
            gamma (float): The discount factor.
            num_episodes (int): The number of episodes to train for."""
        self.alpha = alpha
        self.start_epsilon = epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_runs = num_runs
        self.Q = np.zeros((X_DIM, Y_DIM, NUM_ACTIONS))
        self.path = []
        self.sum_of_rewards_during_episodes = np.zeros(num_episodes)
        self.use_epsilon_decay = use_epsilon_decay
        if self.use_epsilon_decay and self.alpha > 0.05:
            self.alpha = 0.01
        self.decay_rate = decay_rate

    def choose_action(self, state, learning=True):
        """
        Chooses an action for the agent to take derived from the Q-table.
        """
        if np.random.rand() < self.epsilon and learning:
            # Take a random action
            return np.random.randint(NUM_ACTIONS)
        else:
            # If state contains non integer values, show them and report error
            if not isinstance(state[0], int) or not isinstance(state[1], int):
                print("State: {}".format(state))
                raise ValueError("State should be an integer tuple")
            # Take the action with the highest Q-value
            return np.argmax(self.Q[state])

    def epsilon_decay(self, ep):
        if self.use_epsilon_decay:
            self.epsilon = self.start_epsilon * (self.decay_rate ** ep)

    def take_action(self, state, action):
        """
        Takes an action and returns the new state and reward.
        """
        if action == 0:
            # Move right
            new_state = (state[0] + 1, state[1])
        elif action == 1:
            # Move up
            new_state = (state[0], state[1] + 1)
        elif action == 2:
            # Move left
            new_state = (state[0] - 1, state[1])
        elif action == 3:
            # Move down
            new_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Invalid action: {}".format(action))
        # Check if the new state is out of bounds
        if new_state[0] < 0 or new_state[0] >= X_DIM or new_state[1] < 0 or new_state[1] >= Y_DIM:
            # Stay in the same state and get a reward of -1
            return state, -1
        # Check if the new state is in the cliff
        if new_state[1] == 0 and new_state[0] != 0 and new_state[0] != X_DIM - 1:
            # Go back to the start and get a reward of -100
            return (0, 0), -100
        # Otherwise, return the new state and a reward of -1
        return new_state, -1

    def episode_finished(self, state):
        """
        Checks if the episode is finished - i.e. if the agent has reached the goal state or fallen off the cliff.
        """
        x = state[0]
        y = state[1]
        if x == X_DIM - 1 and y == 0:
            # Reached the goal state
            return True
        if x != 0 and x != X_DIM - 1 and y == 0:
            # Fell off the cliff
            return True
        return False

    def learn(self):
        """Trains the agent using the SARSA algorithm."""
        print("Training SARSA agent...")
        for run in trange(self.num_runs):
            self.epsilon = self.start_epsilon
            for episode in range(self.num_episodes):
                S = (0, 0)
                A = self.choose_action(S)
                current_episode_reward_sum = 0
                while not self.episode_finished(S):
                    S_prime, R = self.take_action(S, A)
                    A_prime = self.choose_action(S_prime)
                    self.Q[S][A] += self.alpha * (R + self.gamma * self.Q[S_prime][A_prime] - self.Q[S][A])
                    S = S_prime
                    A = A_prime
                    current_episode_reward_sum += R
                self.sum_of_rewards_during_episodes[episode] += current_episode_reward_sum
                self.epsilon_decay(episode)
        self.sum_of_rewards_during_episodes /= self.num_runs

    def get_path(self, start_state=(0, 0), end_state=(11, 0)):
        """
        Returns a path that the agent takes from the start state to the end state.
        """
        self.path = []
        S = start_state
        self.path.append(S)
        while not self.episode_finished(S):
            A = self.choose_action(S, learning=False)
            S_prime, _ = self.take_action(S, A)
            S = S_prime
            self.path.append(S)
        self.path.append(end_state)
        return self.path

    def get_sum_of_rewards_during_episodes(self):
        """
        Returns the sum of rewards during each episode.
        """
        return self.sum_of_rewards_during_episodes

def plot_gridworld(path1, path2):
    """
    Plots a gridworld for the cliff-walking problem.
    """
    plt.figure()
    plt.gcf().set_size_inches(X_DIM, Y_DIM)
    plt.title("Cliff-Walking Gridworld")
    # Set limits
    plt.xlim(-0.5, X_DIM - 0.5)
    plt.ylim(-0.5, Y_DIM - 0.5)
    # Show ticks at -0.5, 0.5, 1.5, etc.
    plt.xticks(np.arange(-0.5, X_DIM, 1))
    plt.yticks(np.arange(-0.5, Y_DIM, 1))
    # Draw gridlines
    plt.grid(True)
    # Do not show numbers on axes
    plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # Draw the cliff as between x=0.5 and x=10.5, y=-0.5 and y=0.5
    plt.fill_between([0.5, 10.5], [-0.5, -0.5], [0.5, 0.5], color="gray")
    # Plot the paths and show the legend
    plt.plot([x for x, _ in path1], [y for _, y in path1], "r-", label="Q-learning Path")
    plt.plot([x for x, _ in path2], [y for _, y in path2], "b-", label="SARSA Path")
    plt.legend()
    # Put a big S at the start state (0, 0)
    plt.text(0, 0, "S", ha="center", va="center", fontsize=20)
    # Put a big G at the goal state (11, 0)
    plt.text(11, 0, "G", ha="center", va="center", fontsize=20)
    plt.show()

def plot_sum_of_rewards(sum_of_q_learning_rewards_during_episodes, sum_of_sarsa_rewards_during_episodes):
    """
    Plots the sum of rewards during each episode vs the episode number.
    """
    plt.figure()
    plt.title("Sum of Rewards During Each Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Sums")
    plt.plot(sum_of_q_learning_rewards_during_episodes, 'r-', label="Q-learning Rewards")
    plt.plot(sum_of_sarsa_rewards_during_episodes, 'b-', label="SARSA Rewards")
    plt.legend()
    plt.show()

def main(epsilon_decay):
    print("TD Programming Assignment")

    # Train a Q-learning agent
    ql_agent = QLearningAgent(use_epsilon_decay=epsilon_decay)
    ql_agent.learn()

    # Train a SARSA agent
    sarsa_agent = SARSAAgent(use_epsilon_decay=epsilon_decay)
    sarsa_agent.learn()

    # Get Q-learning agent's path
    path1 = ql_agent.get_path()
    path2 = sarsa_agent.get_path()

    plot_gridworld(path1, path2)

    plot_sum_of_rewards(ql_agent.get_sum_of_rewards_during_episodes(), sarsa_agent.get_sum_of_rewards_during_episodes())

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please specify if you want epsilon decay via True or False")
    else:
        main(eval(sys.argv[1]))
