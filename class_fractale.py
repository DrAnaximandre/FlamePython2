from utils import *
import numpy as np
from PIL import Image, ImageFilter
import sys
import matplotlib.pyplot as plt

class Function:
    '''
        Defines a function to be applied to the coordinates.

        Parameters:
            - ws the weights : it's a list of weights that are applied
                to each parameters for each function (additives).
            - params : a list of lenght 6.
                The 3 first are the coefficients for resp. a constant, x,
                and y to form the x of the vector that goes in the additives.
                  Same for [3:6] that forms the y.
            - additives: a list of functions in utils.py.
                So far are implemented: linear, swirl, spherical,
                expinj, pdj, bubble.
    '''

    def __init__(self, ws, params, additives):

        self.ws = ws
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
        for i in range(len(self.ws)):
            res += self.ws[i] * self.additives[i](x_loc, y_loc)
        return res


class Variation:
    '''
        A variation is a set of several functions.

        It takes no parameters to build since they are all
        updated each time you add a function.

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

    def __init__(self):
        self.Nfunctions = 0  # the number of functions in the Variation
        self.functions = []  # a list where the functions are stored
        self.vproba = [0]  # a list of probabilities, see doc
        self.cols = []  # a list of colors associated to each function
        self.lockVariation = False  # a bool: can I still add functions?
        self.rotation = []  # a list of rotations to be applied
        self.final = False  # a bool: does the variation has a final function?

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
            onesfinal = np.ones(Nloc)
            # As said in the Function.call doc, the first column of a
            # batch of points should be full of ones for the call to work.
            resF = self.final.call(np.column_stack((onesfinal, resF)))
        else:
            pass
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


class Fractale:

    def __init__(self, burn, niter, zoom=1):
        self.zoom = zoom
        self.variations = []
        self.Ns = []
        # list that stores the number of points for each variation
        self.burn = burn
        self.niter = niter
        self.lockBuild = False

    def addVariation(self, var, N):
        self.variations.append(var)
        self.Ns.append(N)

    def build(self):
        '''
            it is not advised to add variations after a build

        '''
        if not self.lockBuild:
            totalSize = np.sum(self.Ns) * self.niter
            self.F = np.random.uniform(-1, 1, size=(totalSize, 2))
            self.C = np.ones(shape=(totalSize, 3)) * 255
            [v.fixProba() for v in self.variations]
            self.lockBuild = True
            self.hmv = len(self.variations)

        else:
            print("You have already built this Fractale")

    def run1iter(self, whichiter, burn):
        # safety check
        if not self.lockBuild:
            print("you are trying to run a Fractale not built")
            sys.exit()

        sumNS = np.sum(self.Ns)
        a = sumNS * whichiter
        b = sumNS * (whichiter + 1)
        c = sumNS * (whichiter + 2)
        rangeIdsI = np.arange(a, b)
        if burn:
            rangeIdsO = rangeIdsI
        else:
            rangeIdsO = np.arange(b, c)

        # safety check
        if len(rangeIdsI) != sumNS:
            print("the number of indices provided is different" +
                  "from the number of points in one image")
            sys.exit()

        ones = np.ones(len(rangeIdsI))
        totoF = np.column_stack((ones, self.F[rangeIdsI, :]))
        totoC = self.C[rangeIdsI, :]

        for i in range(self.hmv):
            snsi = sum(self.Ns[:i])
            ids = np.arange(snsi, snsi + self.Ns[i])

            resloc, coloc = self.variations[i].runAllfunctions(
                totoF[ids, :], totoC[ids, :])
            storageF = self.variations[i].runAllrotations(resloc)
            self.F[rangeIdsO[ids], :] = storageF
            self.C[rangeIdsO[ids], :] = coloc

        self.C[rangeIdsO, :] /= 2

    def runAll(self):
        for i in np.arange(self.burn):
            self.run1iter(0, True)
        for i in np.arange(self.niter - 1):
            self.run1iter(i, False)
        self.F = self.F * self.zoom

    def toScore(self, divs=4):
        print("Scoring ... ")
        conditions = np.zeros((self.F.shape[0], 4), dtype='bool')
        conditions[:, 0] = self.F[:, 0] < 1
        conditions[:, 1] = self.F[:, 0] > -1
        conditions[:, 2] = self.F[:, 1] < 1
        conditions[:, 3] = self.F[:, 1] > -1
        goods = np.where(np.all(conditions, 1))[0]
        hscore = depthcut(self.F[goods], 1, -1, -1, 1, 0, divs, [], "start")
        res = ""
        for i in range(divs):
            datlevel = [score[0]
                        for score in hscore if score[1] == i]
            datstring = [str(i) for i in datlevel]
            res += ";".join(datstring)
            res += ";"
        return(res)

    def toImage(self,
                sizeImage=1000,
                coef_forget=1.,
                coef_intensity=.25,
                optional_kernel_filtering=True,
                verbose=0):

        imgtemp = Image.new('RGB', (sizeImage, sizeImage), "black")
        bitmap = np.array(imgtemp)
        intensity = np.zeros((sizeImage, sizeImage, 3))

        F_loc = (sizeImage * (self.F + 1) / 2).astype("i2")

        conditions = np.zeros((F_loc.shape[0], 4), dtype='bool')
        conditions[:, 0] = F_loc[:, 0] < sizeImage
        conditions[:, 1] = F_loc[:, 0] > 0
        conditions[:, 2] = F_loc[:, 1] < sizeImage
        conditions[:, 3] = F_loc[:, 1] > 0

        goods = np.where(np.all(conditions, 1))[0]
        if verbose > 0:
            print("    number of points in the image: " + str(len(goods)))

        bitmap, intensity = renderImage(F_loc, self.C, bitmap,
                                        intensity, goods, coef_forget)

        nmax = np.amax(intensity)
        if verbose > 0:
            print("    nmax: " + str(nmax))
        intensity = np.power(np.log(intensity + 1
                                    ) / np.log(nmax + 1), coef_intensity)

        bitmap = np.uint8(bitmap * intensity)

        out = Image.fromarray(bitmap)

        # Kernel filtering part
        if optional_kernel_filtering:
            # print("    starting Kernel smoothing")
            kfilter = np.ones((3, 3))
            #kfilter[1:4, 1:4] = 2
            #kfilter[2, 2] = 3
            kfilter[1, 1] = 2
            supsammpK = ImageFilter.Kernel((3, 3), kfilter.flatten())
            out = out.filter(supsammpK)

        return(out, bitmap)


if __name__ == '__main__':

    burn = 0
    niter = 25
    zoom = 1
    N = 5000
    end = False
    ci = 0.5
    fi = 1
    colors = [[250, 0, 0],
              [250, 250, 0],
              [0, 250, 0],
              [0, 0, 250],
              [28, 75, 250]]

    A = np.array(np.random.normal(0,2/np.sqrt(6),(6,5)))


    while not end:

        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        for i in range(5):
            v1.addFunction([.5], A[:,i], [linear], 0.2, colors[i])

        F1.addVariation(v1, N)
        F1.build()
        print("Running")
        F1.runAll()
        print("Generating the image")


        out, bitmap = F1.toImage(1024, 
            coef_forget=fi, 
            coef_intensity=ci,
            optional_kernel_filtering=False)
        
        plt.imshow(bitmap, interpolation = 'bicubic')
        plt.show()

        print("--- Actions ---")
        print("Save the image? (S)")
        print("Change intensity coefficient? (c)")
        print("Change forget coefficient? (f)")
        action = input("Your action ?")
        
        if action =="S":
            out.save("test.png")
            end = True
        elif action == "c":
            ci = float(input("Coef intensity ? "))
        elif action == "f":
            fi = float(input("Coef forget ? "))
        else:
            print("please choose again")
            action = input("Your action ?")


    # from helpers import make_serp()
    # make_serp()
    # make_mess()
