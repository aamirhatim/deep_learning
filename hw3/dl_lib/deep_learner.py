import autograd.numpy as np
from dl_lib import multilayer_perceptrons as MLP
from dl_lib import normalizers as Normalizer
from dl_lib import cost_functions as Cost
from dl_lib import optimizers as Optimizer
from dl_lib import plotters as Plotter

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

        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        else:
            self.beta = 0

        self.w0 = self.weight_matrix()                      # Create weight matrix

        # Run gradient descent
        if 'version' in kwargs:
            self.optimizer_version = kwargs['version']
        else:
            self.optimizer_version = 'standard'

        if self.optimizer_version == 'standard':
            weights, costs = Optimizer.gradient_descent(self.cost, self.alpha_choice, self.max_its, self.w0)
            self.weights.append(weights)                        # Save weight history
            self.costs.append(costs)                            # Save cost history
            print('Standard gradient descent with alpha =', self.alpha_choice, 'and beta =', self.beta,'@', self.max_its, 'iterations')
        elif self.optimizer_version == 'normalized':
            weights, costs = Optimizer.normalized_gradient_descent(self.cost, self.alpha_choice, self.max_its, self.w0, self.beta)
            self.weights.append(weights)                        # Save weight history
            self.costs.append(costs)                            # Save cost history
            print('Normalized gradient descent with alpha =', self.alpha_choice, 'and beta =', self.beta,'@', self.max_its, 'iterations')

        if self.cost_name in ['softmax', 'multiclass_softmax']:
            counts = [self.counter(v) for v in weights]
            self.counts.append(counts)                      # Save misclassification history


    def show_history(self):
        '''
        Plot cost and misclassification histories
        '''
        Plotter.Histories(self.costs, self.counts, self.alpha_choice)

    def plot_model(self):
        '''
        Visualization of a best fit model
        '''
        ind = np.argmin(self.costs)
        least_weights = self.weights[0][ind]
        Plotter.Model(self.x, self.y, least_weights, self.costs[0], self.normalizer, self.model)
