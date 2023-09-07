# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np

def bandit_distributions(k=10, size=1000):
    # Create k (default 10) stationary probability distributions (default size 1000) with mean 0 and variance 1.
    # These represent the true values of each action.
    bandits = []
    for i in range(10):
        bandits.append(np.random.normal(0, 1, (1,size)))
    # Show means and variances of each bandit
    for i in range(10):
        mean = np.mean(bandits[i])
        mean_str = ""
        # Format mean to be 7 characters long
        if mean < 0:
            mean_str = str(mean)[:8]
        else:
            mean_str = "+" + str(mean)[:7]
        # Format variance to be 7 characters long
        variance_str = str(np.var(bandits[i]))[:7]
        print(f"Bandit {i}: Mean = {mean_str}, Variance = {variance_str}")

if __name__ == "__main__":
    bandit_distributions()