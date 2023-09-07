# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np
import matplotlib.pyplot as plt

class KArmedBandit:
    def __init__(self, num_arms=10, show_plots=True):
        self.num_arms = num_arms
        self.create_distributions()
        self.shift_distributions()
        self.print_distributions()
        if show_plots:
            self.plot_distributions()

    def create_distributions(self, size=1000):
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

if __name__ == "__main__":
    # Create a 10-armed bandit
    bandit = KArmedBandit(num_arms=10, show_plots=False)