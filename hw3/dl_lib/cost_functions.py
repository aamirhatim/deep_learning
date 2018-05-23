import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self, name, x, y, feature_transforms):
        self.x = x                                                  # Link input and output data
        self.y = y
        self.feature_transforms = feature_transforms
        self.sig = signature(self.feature_transforms)               # Get number of params in feature transform

        if name == 'least_squares':
            self.cost = self.least_squares

        elif name == 'softmax':
            self.cost = self.softmax
        elif name == 'twoclass_counter':
            self.cost = self.two_class_counter

        elif name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        elif name == 'multiclass_counter':
            self.cost = self.multiclass_counter

    def model(self, x, w):
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        # f = 0
        # if len(self.sig.parameters) == 2:
        #     f = self.feature_transforms(x,w[0])
        # else:
        #     f = self.feature_transforms(x)
        #
        # # compute linear combination and return
        # # switch for dealing with feature transforms that either
        # # do or do not have internal parameters
        # a = 0
        # if len(self.sig.parameters) == 2:
        #     a = w[1][0] + np.dot(f.T,w[1][1:])
        # else:
        #     a = w[0] + np.dot(f.T,w[1:])

        f = self.feature_transforms(x,w[0])
        a = w[1][0] + np.dot(f.T,w[1][1:])

        return a.T

    def least_squares(self, w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(np.size(self.y))

    def softmax(self, w):
        cost = np.sum(np.log(1 + np.exp(-self.y*model(self.x,w))))
        return cost/float(np.size(self.y))

    def two_class_counter(w,x):
        misclassification = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*misclassification

    def multiclass_softmax(self, w):
        # get subset of points
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals), axis = 0))

        # compute cost in compact form using numpy broadcasting
        # print(all_evals)
        # print(self.y.astype(int).flatten())
        # print(np.arange(np.size(self.y)))
        b = all_evals[self.y.astype(int).flatten(),np.arange(np.size(self.y))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(self.y))

    def multiclass_counter(self, w):
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        misclassifications = np.sum(np.abs(np.sign(self.y - y_predict)))

        return misclassifications
