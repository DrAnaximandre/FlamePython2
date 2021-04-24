from utils import *
import numpy as np
from PIL import Image, ImageFilter
import sys


class FractalParameters(object):

    def __init__(self):

        self.burn = 5 
        self.niter = 25
        self.zoom = 1


class Fractal:

    def __init__(self, fractal_parameters: FractalParameters):


        self.fractal_parameters = FractalParameters()
        
        # Internals
        self.variations = []
        self.already_built = False
        # self.Ns = []
        # list that stores the number of points for each variation
        

    def addVariation(self, var):
        self.variations.append(var)

    def build(self):
        '''
            it is not advised to add variations after a build
        '''
        if not self.already_built:
            self.sumNs = sum([var.N.N for var in self.variations])
            totalSize = self.sumNs * self.fractal_parameters.niter
            self.F = np.random.uniform(-1, 1, size=(totalSize, 2))
            self.C = np.ones(shape=(totalSize, 3)) * 255
            [v.fixProba() for v in self.variations]
            self.hmv = len(self.variations)
            self.already_built = True

        else:
            print("You have already built this Fractale")

    def run1iter(self, whichiter, burn):
        # safety check
        if not self.already_built:
            print("you are trying to run a Fractale not built")
            sys.exit()

        a = self.sumNs * whichiter
        b = self.sumNs * (whichiter + 1)
        c = self.sumNs * (whichiter + 2)
        rangeIdsI = np.arange(a, b)
        if burn:
            rangeIdsO = rangeIdsI
        else:
            rangeIdsO = np.arange(b, c)

        # safety check
        if len(rangeIdsI) != self.sumNs:
            print("the number of indices provided is different" +
                  "from the number of points in one image")
            sys.exit()

        totoF = self.F[rangeIdsI, :]
        totoC = self.C[rangeIdsI, :]

        for i in range(self.hmv):
            snsi = sum([var.N for var in self.variations[:i]])
            ids = np.arange(snsi, snsi + self.variations[i].N.N)  # ugh

            resloc, coloc = self.variations[i].runAllfunctions(
                totoF[ids, :], totoC[ids, :])
            storageF = self.variations[i].runAllrotations(resloc)
            self.F[rangeIdsO[ids], :] = storageF
            self.C[rangeIdsO[ids], :] = coloc

        self.C[rangeIdsO, :] /= 2

    def runAll(self):
        for i in np.arange(self.fractal_parameters.burn):
            self.run1iter(0, True)
        for i in np.arange(self.fractal_parameters.niter - 1):
            self.run1iter(i, False)
        self.F = self.F * self.fractal_parameters.zoom

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
        bitmap = np.array(imgtemp).astype(np.float)
        intensity = np.zeros((sizeImage, sizeImage))

        F_loc = (sizeImage * (self.F + 1) / 2).astype("i2")

        conditions = np.zeros((F_loc.shape[0], 4), dtype='bool')
        conditions[:, 0] = F_loc[:, 0] < sizeImage
        conditions[:, 1] = F_loc[:, 0] > 0
        conditions[:, 2] = F_loc[:, 1] < sizeImage
        conditions[:, 3] = F_loc[:, 1] > 0

        goods = np.where(np.all(conditions, 1))[0]
        if verbose > 0:
            print("    number of points in the image: " + str(len(goods)))

        bitmap, intensity = renderImage(
            F_loc, self.C, bitmap,intensity, goods, coef_forget)

        nmax = np.amax(intensity)
        if verbose > 0:
            print("    nmax: " + str(nmax))
        intensity = np.power(np.log(intensity + 1
                                    ) / np.log(nmax + 1), coef_intensity)

        bitmap = np.uint8(bitmap * np.reshape(np.repeat(intensity,3), (sizeImage,sizeImage,3)))

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

    from helpers import *
    make_serp()
    # make_mess()
    # make_final()
    # make_quizz()
