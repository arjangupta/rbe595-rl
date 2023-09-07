# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np
import matplotlib.pyplot as plt
import random

class KArmedBandit:
    def __init__(self, num_arms=10, show_plots=True, num_steps=1000):
        self.num_arms = num_arms
        self.create_distributions(size=num_steps)
        self.shift_distributions()
        self.print_distributions()
        if show_plots:
            self.plot_distributions()

    def create_distributions(self, size):
        """Create k (default 10) stationary probability distributions (default size 1000)
           with mean 0 and variance 1. These represent the true values of each action."""
        self.distributions = []
        for i in range(self.num_arms):
            self.distributions.append(np.random.normal(0, 1, (1,size))) # this is a Gaussian distribution

    def print_distributions(self):
        """Show means and variances of each bandit"""
        print("---- True values of each action/arm ----")
        for i in range(self.num_arms):
            mean = np.mean(self.distributions[i])
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
        textbook_shifts = [0.20, -0.80, 1.50, 0.40, 1.05, -1.50, -0.15, -1.00, 1.75, -0.50]
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
        return self.distributions[action][0][current_step]

class ActionValueMethod:
    def __init__(self, bandit: KArmedBandit, epsilon, num_steps=1000):
        self.bandit = bandit
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.estimated_reward = np.zeros(10) #Q
        self.number_of_times_action_taken = np.zeros(10) #N
        self.running_average = np.zeros(num_steps)

    def run(self):

        for n in range(num_steps):
            rand = np.random.uniform(0, 1)
            action = 0

            if rand <= self.epsilon:
                action = np.random.randint(0, 10)
            else:
                action = self.estimated_reward.argmax()

            reward = self.bandit.take_action(action, n)

            self.number_of_times_action_taken[action] = self.number_of_times_action_taken[action] + 1

            previous_reward = self.estimated_reward[action]

            self.estimated_reward[action] = previous_reward + (1/self.number_of_times_action_taken[action])*(reward - previous_reward)

            # Record the current running average
            self.running_average[n] = np.average(self.estimated_reward)

def plot(running_average1, running_average2, running_average3):
    """Plot three graphs with various epsilon values"""

    plt.plot(running_average1)
    plt.plot(running_average2)
    plt.plot(running_average3)
    plt.xlabel("Steps")
    plt.ylabel("Estimated Reward")
    plt.title("Estimated Reward vs Steps")
    # Add legend
    plt.legend(["Epsilon = 0", "Epsilon = 0.1", "Epsilon = 0.01"])
    plt.show()


if __name__ == "__main__":
    # Set the total steps for each run
    num_steps = 1000
    # Create a 10-armed bandit
    bandit = KArmedBandit(num_arms=10, show_plots=False, num_steps=num_steps)
    # Run the action-value method with epsilon = 0 (greedy only)
    greedy_method = ActionValueMethod(bandit, epsilon=0, num_steps=num_steps)
    greedy_method.run()
    # Run the action-value method with epsilon = 0.1 and epsilon = 0.01
    epsilon_greedy1 = ActionValueMethod(bandit, epsilon=0.1, num_steps=num_steps)
    epsilon_greedy1.run()
    epsilon_greedy2 = ActionValueMethod(bandit, epsilon=0.01, num_steps=num_steps)
    epsilon_greedy2.run()
    plot(greedy_method.running_average, epsilon_greedy1.running_average, epsilon_greedy2.running_average)