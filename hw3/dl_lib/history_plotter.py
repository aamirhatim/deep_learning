import autograd.numpy as np
import matplotlib.pyplot as plt

class Histories:
    def __init__(self, costs, counts, alpha):
        if len(counts) == 0:
            self.plot_cost(costs, alpha)                              # Only plot costs if counts is empty
        else:
            self.plot_cost_counts(costs, counts, alpha)               # Else plot cost and counts history

    def plot_cost(self, costs, alpha):
        plt.figure(figsize = (15,5))
        plt.suptitle('Results (step size = %i)' %alpha)
        plt.plot(np.linspace(1,len(costs[0])+1,len(costs[0])+1), costs[0])
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend(title = "Alpha", loc = "upper right")
        plt.show

    def plot_cost_counts(self, costs, counts, alpha):
        plt.figure(figsize = (15,5))
        plt.suptitle('Results (step size = %i)' %alpha)
        plt.subplot(1,2,1)
        plt.plot(np.linspace(1,len(costs[0]),len(costs[0])), costs[0])
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")

        plt.subplot(1,2,2)
        plt.plot(np.linspace(1,len(counts[0]),len(counts[0])), counts[0])
        plt.title("Misclassification History")
        plt.xlabel("Iteration")
        plt.ylabel("Misclassifications")
        plt.show

class Model:
    def __init__(self, x, y, weights, model):
        self.x = x
        self.y = y

        print(x)
        print(y)
        y_model = model(x, weights)
        print(y_model)
