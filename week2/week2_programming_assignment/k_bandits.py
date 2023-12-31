# Authors: Taylor Bergeron and Arjan Gupta
# RBE 595 Reinforcement Learning
# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np
import matplotlib.pyplot as plt
import sys

class KArmedBandit:
    def __init__(self, num_arms=10, show_plots=True, num_steps=1000):
        self.num_arms = num_arms
        self.create_distributions(size=num_steps)
        self.shift_distributions()
        self.find_means()
        self.find_optimal_action()
        if show_plots:
            self.print_distributions()
            self.plot_distributions()

    def create_distributions(self, size):
        """Create k (default 10) stationary probability distributions (default size 1000)
           with mean 0 and variance 1. These represent the true values of each action."""
        self.distributions = []
        for i in range(self.num_arms):
            # For each arm generate a normal distribution with mean 0 and variance 1 (std dev 1)
            self.distributions.append(np.random.normal(0, 1, size))

    def find_means(self):
        """Get the means of each bandit"""
        self.means = []
        for i in range(self.num_arms):
            self.means.append(np.mean(self.distributions[i]))

    def find_optimal_action(self):
        """Fix the optimal action based on the mean of each distribution"""
        self.optimal_action = np.argmax(self.means)

    def print_distributions(self):
        """Show means and variances of each bandit"""
        print("---- True values of each action/arm ----")
        for i in range(self.num_arms):
            mean = self.means[i]
            mean_str = ""
            # Format mean to be 7 characters long
            if mean < 0:
                mean_str = str(mean)[:8]
            else:
                mean_str = "+" + str(mean)[:7]
            # Format variance to be 7 characters long
            variance_str = str(np.var(self.distributions[i]))[:7]
            if i < 9:
                print(f"Arm {i+1}:  Mean = {mean_str}, Variance = {variance_str}")
            else:
                print(f"Arm {i+1}: Mean = {mean_str}, Variance = {variance_str}")

    def shift_distributions(self):
        """Shift each bandit's distribution by roughly the amounts shown in the textbook."""
        #                   1      2      3     4     5      6     7      8     9      10
        textbook_shifts = [0.20, -0.80, 1.50, 0.40, 1.05, -1.50, -0.15, -1.00, 0.75, -0.50]
        for i in range(self.num_arms):
            self.distributions[i] += textbook_shifts[i]

    def plot_distributions(self):
        """Plot the distributions of each bandit:
           - Plot each bandit's distribution in a subplot
           - Label each subplot with the bandit's number
           - Mark the mean of each distribution with a red vertical line
           """
        fig, axs = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle("True distributions of each arm (red line is mean)")
        for i in range(self.num_arms):
            row = i // 5
            col = i % 5
            axs[row, col].hist(self.distributions[i][0], bins=50)
            # Show red line and show its value in the x-axis
            axs[row, col].axvline(np.mean(self.distributions[i]), color='r')
            axs[row, col].text(np.mean(self.distributions[i]), 0, f"{np.mean(self.distributions[i]):.2f}", color='r')
            # Set title, x-axis label, and y-axis label
            axs[row, col].set_title(f"Bandit {i+1}")
            axs[row, col].set_xlabel("Reward distribution")
            axs[row, col].set_ylabel("Frequency")
        plt.show()
    
    def take_action(self, action, current_step):
        """Take an action and return a reward"""
        return self.distributions[action][current_step]

class ActionValueMethod:
    """Class to perform the action-value method for a given bandit and epsilon value"""
    def __init__(self, bandit: KArmedBandit, epsilon, num_steps=1000):
        self.bandit = bandit
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.estimated_reward = np.zeros(10) # this is Q as described in the textbook
        self.number_of_times_action_taken = np.zeros(10) # this is N as described in the textbook
        self.rewards = np.zeros(num_steps)
        self.optimal_actions = np.zeros(num_steps)

    def run(self):
        """Run the action-value method for given number of steps. Follows the pseudocode given in the textbook.
           Returns the average reward for each step as well as the optimal actions for each run"""
        for n in range(self.num_steps):
            exploration_decision = np.random.uniform(0, 1)
            action = 0
            # Based on the epsilon value, either take a random action (which is exploration)
            # or take the action with the highest estimated reward (which is the greedy strategy)
            if exploration_decision <= self.epsilon:
                action = np.random.randint(0, 10)
            else:
                # Find the max arg of the estimated reward
                action = self.estimated_reward.argmax()
                # If this estimation occurs more than once, randomly choose one of the actions
                action = np.random.choice(np.where(self.estimated_reward == self.estimated_reward[action])[0])

            # Take the action and get the reward (this is R as described in the textbook)
            reward = self.bandit.take_action(action, n)

            # Increment the number of times this action was taken
            self.number_of_times_action_taken[action] = self.number_of_times_action_taken[action] + 1

            # Update the estimated reward for this action
            previous_reward = self.estimated_reward[action]
            self.estimated_reward[action] = previous_reward + (1/self.number_of_times_action_taken[action])*(reward - previous_reward)

            # Record the current reward
            self.rewards[n] = reward

            # Record whether the action taken was optimal
            self.optimal_actions[n] = self.bandit.optimal_action == action

        return self.rewards, self.optimal_actions

def plot_graph1(average_rewards1, average_rewards2, average_rewards3, num_runs):
    """Plot three graphs of average rewards with various epsilon values"""
    plt.plot(average_rewards1, color='green')
    plt.plot(average_rewards2, color='blue')
    plt.plot(average_rewards3, color='red')
    plt.xlabel("Steps")
    plt.ylabel(f"Average reward over {num_runs} runs")
    plt.title("Average Reward vs Steps")
    # Add legend
    plt.legend(["Epsilon = 0", "Epsilon = 0.1", "Epsilon = 0.01"])
    plt.show()

def plot_graph2(optimal_actions1, optimal_actions2, optimal_actions3, num_runs):
    """Plot three graphs of optimal actions with various epsilon values"""
    plt.plot(optimal_actions1, color='green')
    plt.plot(optimal_actions2, color='blue')
    plt.plot(optimal_actions3, color='red')
    plt.xlabel("Steps")
    plt.ylabel(f"Optimal action over {num_runs} runs")
    plt.title("Optimal Action vs Steps")
    # Add legend
    plt.legend(["Epsilon = 0", "Epsilon = 0.1", "Epsilon = 0.01"])
    # Have y-axis show as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}%".format(int(x * 100))))
    plt.show()

def get_graphs(num_runs = 2000, print_distribution_graphs = False):
    """Obtain the two graphs for the exercise"""

    # Set the total steps for each run
    num_steps = 1000
    # Declare arrays to store the average rewards for each method
    greedy_rewards = np.zeros(num_steps)
    epsilon_greedy_1_rewards = np.zeros(num_steps)
    epsilon_greedy_2_rewards = np.zeros(num_steps)
    # Declare arrays to store the optimal actions for each method
    greedy_optimal_actions = np.zeros(num_steps)
    epsilon_greedy_1_optimal_actions = np.zeros(num_steps)
    epsilon_greedy_2_optimal_actions = np.zeros(num_steps)

    # Perform the given number of runs
    for run in range(num_runs):
        print(f"Performing run {run+1} of {num_runs}...")
        # Create a 10-armed bandit
        bandit = KArmedBandit(num_arms=10, show_plots=print_distribution_graphs, num_steps=num_steps)

        # Run the action-value method with epsilon = 0 (greedy only)
        greedy_method = ActionValueMethod(bandit, epsilon=0, num_steps=num_steps)
        greedy_run_results = greedy_method.run()
        greedy_rewards = np.add(greedy_rewards, greedy_run_results[0])
        greedy_optimal_actions = np.add(greedy_optimal_actions, greedy_run_results[1])

        # Run the action-value method with epsilon = 0.1
        epsilon_greedy1 = ActionValueMethod(bandit, epsilon=0.1, num_steps=num_steps)
        epsilon_greedy_1_run_results = epsilon_greedy1.run()
        epsilon_greedy_1_rewards = np.add(epsilon_greedy_1_rewards, epsilon_greedy_1_run_results[0])
        epsilon_greedy_1_optimal_actions = np.add(epsilon_greedy_1_optimal_actions, epsilon_greedy_1_run_results[1])

        # Run the action-value method with epsilon = 0.01
        epsilon_greedy2 = ActionValueMethod(bandit, epsilon=0.01, num_steps=num_steps)
        epsilon_greedy_2_run_results = epsilon_greedy2.run()
        epsilon_greedy_2_rewards = np.add(epsilon_greedy_2_rewards, epsilon_greedy_2_run_results[0])
        epsilon_greedy_2_optimal_actions = np.add(epsilon_greedy_2_optimal_actions, epsilon_greedy_2_run_results[1])
    print("Done all runs!")
    # Plot the average rewards for each method
    plot_graph1(greedy_rewards/num_runs, epsilon_greedy_1_rewards/num_runs, epsilon_greedy_2_rewards/num_runs, num_runs)
    # Plot the optimal actions for each method
    plot_graph2(greedy_optimal_actions/num_runs, epsilon_greedy_1_optimal_actions/num_runs, epsilon_greedy_2_optimal_actions/num_runs, num_runs)

if __name__ == "__main__":
    # If user input was given, use that as the number of runs
    if len(sys.argv) == 2:
        num_runs = int(sys.argv[1])
        get_graphs(num_runs)
    elif len(sys.argv) > 2:
        num_runs = int(sys.argv[1])
        print_distribution_graphs = sys.argv[2].lower() == 'true'
        get_graphs(num_runs, print_distribution_graphs)
    else:
        get_graphs()


