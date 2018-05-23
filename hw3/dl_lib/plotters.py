import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
    def __init__(self, x, y, w, model, **kwargs):
        # fit_points = np.array([np.linspace(min(x[0]),max(x[0]), 100)])
        # y_model = model(fit_points, min_weights)             # Get output from final model
        # self.plot_fit(s, t)
        # print(s)
        # print(t)
        self.x = x
        self.y = y
        self.plot_fit(w, model, **kwargs)

    def plot_fit(self, w, model, **kwargs):
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(9,4))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1])
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax = plt.subplot(gs[1]);
        ax3 = plt.subplot(gs[2]); ax3.axis('off')

        # scatter points
        xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # label axes
        ax.set_xlabel(r'$x$', fontsize = 16)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)

        # create fit
        s = np.linspace(xmin,xmax,300)[np.newaxis,:]
        colors = ['k','magenta']
        if 'colors' in kwargs:
            colors = kwargs['colors']
        c = 0

        normalizer = lambda a: a
        if 'normalizer' in kwargs:
            normalizer = kwargs['normalizer']

        t = model(normalizer(s),w)
        ax.plot(s.T,t.T,linewidth = 4,c = 'k')
        ax.plot(s.T,t.T,linewidth = 2,c = 'r')
