import numpy as np
import matplotlib.pyplot as plt

class DynamicPlot:
    def __init__(self, title='Dynamic Plot', xlabel='X', ylabel='Y'):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def update(self, x, y):
        # Clear the previous plot
        self.ax.clear()
        
        # Plot the current state of the list
        self.ax.plot(x, y, marker='o')
        
        # Add titles and labels
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        # Draw the plot
        plt.draw()
    
    