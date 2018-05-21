import autograd.numpy as np

class Setup:
    def __init__(self, **kwargs):
        a = 'relu'                                              # Set default activation to ReLU
        if 'activation' in kwargs:
            a = kwargs['activation']                            # Manually set activation if present

        if a == 'linear':                                       # Preset activations
            self.activation = lambda data: data
        elif a == 'relu':
            self.activation = lambda data: np.maximum(0, data)
        elif a == 'tanh':
            self.activation = lambda data: np.tanh(data)
        elif a == 'sin':
            self.activation = lambda data: np.sin(data)
        elif a == 'sinc':
            self.activation = lambda data: np.sinc(data)
        elif a == 'maxout':
            self.activation = lambda data1, data2: np.maximum(data1, data2)
            self.weight_matrix = self.maxout_init_weights
            self.transforms = self.maxout_feature_transforms
        else:                                                   # Manual activation
            self.activation = kwargs['activation']

        if self.activation in ['linear', 'relu', 'tanh', 'sin', 'sinc']:
            self.weight_matrix = self.init_weights
            self.transforms = self.feature_transforms

        if 'layer_sizes' in kwargs:
            self.layer_sizes = kwargs['layer_sizes']            # Set layer sizes
        else:                                                   # Else create default setup
            N = 1                                               # Input dimensions
            M = 1                                               # Output dimensions
            U = 10                                              # 10-unit hidden layer
            self.layer_sizes = [N, U, M]                        # Build layer sizes to generate weight matrix

        if 'scale' in kwargs:
            self.scale = kwargs['scale']                        # Set scale
        else:
            self.scale = 0.1

    def init_weights(self):
        weights = []                                        # Container for weights

        # Loop over desired layer sizes and create
        # appropriately sized initial weight matrix
        # for each layer.
        for k in range(len(self.layer_sizes)-1):
            U_k = self.layer_sizes[k]                       # get layer sizes for current weight matrix
            U_k_plus_1 = self.layer_sizes[k+1]

            # make weight matrix
            weight = self.scale*np.random.randn(U_k+1, U_k_plus_1)
            weights.append(weight)

        # Re-express weights so that w_init[0] = omega_inner contains all
        # internal weight matrices, and w_init = w contains weights of
        # final linear combination in predict function.
        w_init = [weights[:-1],weights[-1]]

        return w_init

    def maxout_init_weights(self):
        # Container for entire weight tensor
        weights = []

        # Loop over desired layer sizes and create appropriately sized initial
        # weight matrix for each layer.
        for k in range(len(self.layer_sizes)-1):
            # Get layer sizes for current weight matrix
            U_k = self.layer_sizes[k]
            U_k_plus_1 = self.layer_sizes[k+1]

            # Make weight matrix
            weight1 = self.scale*np.random.randn(U_k + 1,U_k_plus_1)

            # Add second matrix for inner weights
            if k < len(self.layer_sizes)-2:
                weight2 = self.scale*np.random.randn(U_k + 1,U_k_plus_1)
                weights.append([weight1,weight2])
            else:
                weights.append(weight1)

        # Re-express weights so that w_init[0] = omega_inner contains all
        # internal weight matrices, and w_init = w contains weights of
        # final linear combination in predict function.
        w_init = [weights[:-1],weights[-1]]

        return w_init

    def feature_transforms(self, a, w):
        for W in w:                                             # Loop through each layer matrix
            a = W[0] + np.dot(a.T, W[1:])                       # Compute inner product with current layer weights
            a = self.activation(a).T                            # Output of layer activation
        return a

    def maxout_feature_transforms(self, a, w):
        for W1,W2 in w:                                         # loop through each layer matrix
            a1 = W1[0][:,np.newaxis] + np.dot(a.T, W1[1:]).T    # compute inner product with current layer weights
            a2 = W2[0][:,np.newaxis] + np.dot(a.T, W2[1:]).T
            a = self.activation(a1,a2)                          # output of layer activation
        return a
