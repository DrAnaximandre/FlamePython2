import numpy as np
from Function import Function
from utils import rotation


class VariationParameters(object):
    
    def __init__(self):

        self.N = 5000

    def __str__(self):

        return(f"{self.N}")

class Variation:
    '''
        A variation is a set of several Functions.

        The basic life cycle of such an object is: init,
        add some functions, then fixProba. The user should not
        play with these objects, they are used in the Fractale class.

        The variation may have a final function that is applied after the
        regular functions.

        The variation can also have rotations that are applied after all the
        functions.

        The Variation has an vproba attribute: before the fixProba, it's
        the weight of each function in the Variation. After the fix, it's
        the cumulative probability to have each of the functions.
    '''

    def __init__(self, N: VariationParameters):
        self.Nfunctions = 0  # the number of functions in the Variation
        self.functions = []  # a list where the functions are stored
        self.vproba = [0]  # a list of probabilities, see doc
        self.cols = []  # a list of colors associated to each function
        self.lockVariation = False  # a bool: can I still add functions?
        self.rotation = []  # a list of rotations to be applied
        self.final = False  # a bool: does the variation has a final function?
        self.N = N  # the number of MC samples run by this Variation

    def addFunction(self, ws, params, additives, proba, col):
        """ adds a function where the parameters are all provided
            - ws, params, additives are parameters that go directely
                in a new Function object
            - proba is a number, the weight of the added function.
                should not be negative.
            - col is the color of the function,
                it is a np array of shape (3,), that should range from 0 to 255

        """
        if proba < 0:
            raise ValueError("no negative weights allowed")

        if not self.lockVariation:
            self.Nfunctions += 1
            self.cols.append(col)
            self.vproba.append(proba)
            self.functions.append(Function(ws, params, additives))
        else:
            raise ValueError(
                "This variation is locked, I cannot add the function")

    def fixProba(self):
        """ Utility function: scales the weights to a cumsum between 0 and 1.
        """
        self.vproba = [p / np.sum(self.vproba) for p in self.vproba]
        self.vproba = np.cumsum(self.vproba)
        # as a result, the last value of vproba is 1
        self.lockVariation = True

    def addFinal(self, ws, params, additives):
        """ adds a final function to the Variation.
            - ws, params, additives are parameters that go directely
                in a new Function object
        """
        if not self.lockVariation:
            self.final = Function(ws, params, additives)
        else:
            raise ValueError(
                "This variation is locked, I cannot add the final")

    def addRotation(self, angle):
        """ adds a rotation to the variation.
           so far only 3 angles are supported : 180, 120 and 90.
        """
        if not self.lockVariation:
            self.rotation.append(angle)
        else:
            raise ValueError(
                "This variation is locked, I cannot add the rotation")

    def runAllfunctions(self, coordinates, batchpointsC):
        """ Calls all the functions (including the final if it exists).

            Parameters:

            - coordinates: np.array of size Number of points x 3
                it's a columns of ones and the coordinates of the points.
            - batchpointsC: np.array of size Number of points x 3
                it's the colors of the points.
                it should scale between 0 and 255

        """
        Nloc = coordinates.shape[0]  # how many points in the batch
        r = np.random.uniform(size=Nloc)  # each point is attributed a rand
        resF = np.zeros(shape=(Nloc, 2))  # creation of the empty results
        resC = np.zeros(shape=(Nloc, 3))

        for i in range(len(self.vproba) - 1):  # for each regular function
            # we select via a mask the points that are attributed a given
            # function
            mask1 = r > self.vproba[i]
            mask2 = r < self.vproba[i + 1]
            sel = np.where((mask1) & (mask2))[0]
            # then we call the function on the slice. The whole process could
            # be parallelized since we work on slices, but it's quite quick
            resF[sel, :] = self.functions[i].call(coordinates[sel, :])
            # then we blend the color of the points with the color of the
            # function by averaging them
            resC[sel, :] = batchpointsC[sel, :] + self.cols[i]

        if self.final:
            # if the variation has a final function, it is applied on
            # every point of resF.
            # note that the final function has no color thus it does not modify
            # resC.
            resF = self.final.call(resF)
        return(resF, resC)

    def runAllrotations(self, resF):
        Nloc = resF.shape[0]
        r = np.random.uniform(size=Nloc)
        for i in range(len(self.rotation)):
            if self.rotation[i] == 120:
                a120 = np.pi * 2 / 3
                resF = rotation(3, a120, resF, r)

            elif self.rotation[i] == 180:
                a180 = np.pi
                resF = rotation(2, a180, resF, r)

            elif self.rotation[i] == 90:
                a90 = np.pi / 2
                resF = rotation(4, a90, resF, r)

            elif type(self.rotation[i]) == tuple:
                ncustom = self.rotation[i][0]
                acustom = float(self.rotation[i][1])
                coef = float(self.rotation[i][2])
                resF = rotation(ncustom, acustom, resF, r, coef)
        return (resF)
