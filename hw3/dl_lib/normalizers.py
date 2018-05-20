import autograd.numpy as np

class Setup:
    def __init__(self, x, name):
        # Standar normalize input
        if name == 'standard':
            self.normalizer, self.inverse_normalizer = self.standard_normalizer(x)
            
        # Else use raw data values
        else:
            self.normalizer = lambda data: data
            self.inverse_normalizer = lambda data: data

    def standard_normalizer(self, x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        return normalizer,inverse_normalizer
