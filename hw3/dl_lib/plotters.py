import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

class Histories:
    def __init__(self, costs, counts, alpha):
        if len(counts) == 0:
            self.plot_cost(costs, alpha)                              # Only plot costs if counts is empty
        else:
            self.plot_cost_counts(costs, counts, alpha)               # Else plot cost and counts history

    def plot_cost(self, costs, alpha):
        plt.figure(figsize = (15,5))
        plt.suptitle('Results (step size = %f)' %alpha)
        plt.plot(np.linspace(1,len(costs[0])+1,len(costs[0])), costs[0])
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

    def plot_cost_counts(self, costs, counts, alpha):
        plt.figure(figsize = (15,5))
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
        plt.suptitle('Results (step size = %f)' %alpha)
        plt.show()

class Model:
    def __init__(self, x, y, w, cost_history, normalizer, model):
        # construct figure
        fig = plt.figure( figsize=(15,5))
        fig.set_tight_layout(False)

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0]);
        ax = plt.subplot(gs[1]);

        # scatter points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(x,y,ax)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('tuned model')

        # create fit
        s = np.linspace(xmin,xmax,300)[np.newaxis,:]
        colors = ['k','magenta']

        c = 0

        #normalizer = lambda a: a
        #if 'normalizer' in kwargs:
        #   normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        #ax.plot(s.T,t.T,linewidth = 4,c = 'k')
        ax.plot(s.T,t.T,linewidth = 2,c = 'r')
        ax1.plot(np.arange(0,len(cost_history)),cost_history,label='alpha= 1')
        ax1.set_title('Cost history')
        ax1.set_ylabel('Cost History')
        ax1.set_xlabel('iteration')
        plt.show()

    def scatter_pts_2d(self,x,y,ax):
        # set plotting limits
        xmax = copy.deepcopy(np.max(x))
        xmin = copy.deepcopy(np.min(x))
        xgap = (xmax - xmin)*0.2
        xmin -= xgap
        xmax += xgap

        ymax = copy.deepcopy(np.max(y))
        ymin = copy.deepcopy(np.min(y))
        ygap = (ymax - ymin)*0.2
        ymin -= ygap
        ymax += ygap

        # initialize points
        ax.scatter(x.flatten(),y.flatten(),color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        return xmin,xmax,ymin,ymax
