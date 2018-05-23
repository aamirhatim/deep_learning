import autograd.numpy as np
from dl_lib import multilayer_perceptrons as MLP
from dl_lib import normalizers as Normalizer
from dl_lib import cost_functions as Cost
from dl_lib import optimizers as Optimizer
from dl_lib import plotters as Plotter

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
from autograd import hessian as compute_hess
import math
import time
from matplotlib import gridspec
import copy
from inspect import signature
from matplotlib.ticker import FormatStrFormatter

class Setup:
    def __init__(self, x, y, **kwargs):
        self.x = x                                          # Link input and output data
        self.y = y

        self.weights = []                                   # Weight histories
        self.costs = []                                     # Cost histories
        self.counts = []                                    # Misclassification histories

    def choose_normalizer(self, name):
        '''
        Initialize normalizer for input data
        '''
        norm = Normalizer.Setup(self.x, name)
        self.normalizer = norm.normalizer
        self.inverse_normalizer = norm.inverse_normalizer
        self.normalizer_name = name

        self.x = self.normalizer(self.x)                    # Normalize input data
        print('Data Normalized as:', self.normalizer_name)

    def choose_features(self, name, **kwargs):
        '''
        Define feature transforms
        '''
        self.feature_name = name
        if name == 'multilayer_perceptron':
            mlp = MLP.Setup(**kwargs)
            self.feature_transforms = mlp.transforms
            self.weight_matrix = mlp.weight_matrix
            self.layer_sizes = mlp.layer_sizes
            self.activation_name = mlp.activation_name
        print('Feature transform:', self.feature_name)
        print('Activation:', self.activation_name)
        print('Layer sizes:', self.layer_sizes)

    def choose_cost(self, name, **kwargs):
        '''
        Choose cost function
        '''
        c = Cost.Setup(name, self.x, self.y, self.feature_transforms)
        self.cost = c.cost
        self.model = c.model
        self.cost_name = name
        print('Cost function set to:', self.cost_name)

        if name == 'softmax':
            c = Cost.Setup('twoclass_counter', self.x, self.y, self.feature_transforms)
            self.counter = c.cost
            self.counter_name = 'twoclass_counter'
            print('Using counter:', self.counter_name)
        elif name == 'multiclass_softmax':
            c = Cost.Setup('multiclass_counter', self.x, self.y, self.feature_transforms)
            self.counter = c.cost
            self.counter_name = 'multiclass_counter'
            print('Using counter:', self.counter_name)


    def optimize(self, **kwargs):
        '''
        Find a classifier that best fits to the data
        '''
        if 'max_its' in kwargs:                             # Get manual param, else set default value
            self.max_its = kwargs['max_its']
        else:
            self.max_its = 500

        if 'alpha_choice' in kwargs:                        # Get manual param, else set default value
            self.alpha_choice = 10**(-kwargs['alpha_choice'])
        else:
            self.alpha_choice = 10**(-1)

        self.w0 = self.weight_matrix()                      # Create weight matrix

        # Run gradient descent
        weights, costs = Optimizer.gradient_descent(self.cost, self.alpha_choice, self.max_its, self.w0)
        self.weights.append(weights)                        # Save weight history
        self.costs.append(costs)                            # Save cost history

        if self.cost_name in ['softmax', 'multiclass_softmax']:
            counts = [self.counter(v) for v in weights]
            self.counts.append(counts)                      # Save misclassification history

        print('Optimized with alpha =', self.alpha_choice, '@', self.max_its, 'iterations')

    def show_history(self):
        '''
        Plot cost and misclassification histories
        '''
        Plotter.Histories(self.costs, self.counts, self.alpha_choice)

    def plot_model(self, w, model, **kwargs):
        '''
        Visualization of a best fit model
        '''
        # construct figure
        fig, axs = plt.subplots(1, 3, figsize=(15,15))

        # create subplot with 2 panels
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1])
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax = plt.subplot(gs[1]);
        ax3 = plt.subplot(gs[2]); ax3.axis('off')

        # # scatter points
        # xmin,xmax,ymin,ymax = self.scatter_pts_2d(self.x,ax)
        #
        # # clean up panel
        # ax.set_xlim([xmin,xmax])
        # ax.set_ylim([ymin,ymax])
        xmin = min(self.x[0])
        xmax = max(self.x[0])

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
        ax.plot(self.x, self.y)



        # History.Model(self.inverse_normalizer(self.x), self.y, min_weights, self.model, self.normalizer, self.inverse_normalizer)
