# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np
import matplotlib.pyplot as plt

class KArmedBandit:
    def __init__(self, num_arms=10):
        self.num_arms = num_arms
        self.create_distributions()
        self.shift_distributions()
        self.print_distributions()
        self.plot_distributions()

    def create_distributions(self, size=1000):
        """Create k (default 10) stationary probability distributions (default size 1000)
           with mean 0 and variance 1. These represent the true values of each action."""
        self.distributions = []
        for i in range(self.num_arms):
            self.distributions.append(np.random.normal(0, 1, (1,size))) # this is a Gaussian distribution

    def print_distributions(self):
        """Show means and variances of each bandit"""
        print("---- True values of each action ----")
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
                print(f"Bandit {i+1}:  Mean = {mean_str}, Variance = {variance_str}")
            else:
                print(f"Bandit {i+1}: Mean = {mean_str}, Variance = {variance_str}")

    def shift_distributions(self):
        """Shift each bandit's distribution by roughly the amounts shown in the textbook."""
        #                   1      2      3     4     5      6     7      8     9      10
        textbook_shifts = [0.20, -0.80, 1.50, 0.40, 1.05, -1.50, -0.15, -1.00, 1.75, -0.50]
        for i in range(self.num_arms):
            self.distributions[i] += textbook_shifts[i]

    def plot_distributions(self):
        """Plot the distributions of each bandit"""
        for i in range(self.num_arms):
            plt.hist(self.distributions[i], bins=100)
            plt.show()

if __name__ == "__main__":
    # Create a 10-armed bandit
    bandit = KArmedBandit(num_arms=10)