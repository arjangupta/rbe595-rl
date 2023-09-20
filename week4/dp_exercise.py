# Program that takes in a 2D array of 0s and 1s and converts
# it to a plot of occupied and unoccupied spaces.

# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Defining the function that will convert the 2D array to a plot
def plot_2d_array(array):
    # Creating a figure and axes
    fig, ax = plt.subplots()
    # Creating a plot of the array
    ax.imshow(array, cmap = 'binary')
    # Displaying the plot
    plt.show()

# Defining the main function
def main():
    # Creating a 2D array of 0s and 1s
    array = np.array([[0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                      [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                      [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
    # Calling the function that will convert the 2D array to a plot
    plot_2d_array(array)
    
# Calling the main function
if __name__ == "__main__":
    main()