import numpy as np


class Function:
    '''
        Defines a function to be applied to the coordinates.

        Parameters:
            - weights : it's a list of weights that are applied
                to each parameters for each function (additives).
            - params : a list of lenght 6.
                The 3 first are the coefficients for resp. a constant, x,
                and y to form the x of the vector that goes in the additives.
                  Same for [3:6] that forms the y.
            - additives: a list of functions
    '''

    def __init__(self, weights, params, additives):

        self.weights = weights
        self.params = params
        self.additives = additives

    def call(self, points):
        ''' Applies the function to a bunch of points.

        Parameters:
            - points is a np.array of size number of points x 3
            technically the first column should be full of ones,
            but it's not checked for performance.
       '''
        x_loc = np.dot(points, self.params[:3])
        y_loc = np.dot(points, self.params[3:])
        res = np.zeros((points.shape[0], 2))
        for i in range(len(self.weights)):
            res += self.weights[i] * self.additives[i](x_loc, y_loc)
        return res