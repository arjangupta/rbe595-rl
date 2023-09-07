# Week 2 Programming Exercise
# Write a program to generate figure 2.2 of the textbook

import numpy as np

# First we set up 10 probability distributions for the 10 bandits, such that
# the mean of each distribution is a random number between 0 and 1. This must
# be a gaussian distribution, so we use np.random.randn to generate the
# distributions.
bandit_means = np.random.randn(10)
print(bandit_means)