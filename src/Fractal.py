from VariationHolder import VariationHolder
from utils import rotation, renderImage
import numpy as np
from PIL import Image, ImageFilter
from typing import List


class FractalParameters(object):

    def __init__(self, burn=5, niter=25, zoom=1, dx=0.0, dy=0.0, final_rot=np.pi/2, verbose=False):

        self.burn = burn 
        self.niter = niter
        self.zoom = zoom
        self.dx = dx
        self.dy = dy
        self.final_rot = final_rot
        self.verbose = verbose

class Fractal:

    def __init__(self,
                 fractal_parameters: FractalParameters,
                 variationholder: VariationHolder):

        # Controllable parameters
        self.fractal_parameters = fractal_parameters
        # Variations
        self.vh = variationholder
        if fractal_parameters.verbose:
            print(self.vh)

        # former build
        self.sumNs = sum([var.N for var in self.vh])
        if fractal_parameters.verbose:
            print(self.sumNs)
        total_number_of_rows = self.sumNs * self.fractal_parameters.niter
        if fractal_parameters.verbose:
            print(total_number_of_rows)
        self.F = np.random.uniform(-1, 1, size=(total_number_of_rows, 2))
        self.C = np.ones(shape=(total_number_of_rows, 3)) * 255


        [v.variation.fixProba() for v in self.vh]
        self.hmv = len(self.vh)

    def run1iter(self, whichiter, burn):

        a = self.sumNs * whichiter
        b = self.sumNs * (whichiter + 1)
        c = self.sumNs * (whichiter + 2)
        rangeIdsI = np.arange(a, b)
        if burn:
            rangeIdsO = rangeIdsI
        else:
            rangeIdsO = np.arange(b, c)

        # print(f"rangeIdsI - {rangeIdsI}")
        totoF = self.F[rangeIdsI, :]
        totoC = self.C[rangeIdsI, :]

        # print(f"totof - {totoF}")

        for i in range(self.hmv):
            snsi = sum([var.N for var in self.vh[:i]])
            # print(f"snsi - {snsi}")
            # print(f"self.variations[i].N - {self.variations[i].N}")
            ids = np.arange(snsi, snsi + self.vh[i].N)  # ugh
           # print(ids)

           # print(f"totoF[ids, :] before - {totoF[ids, :]}")
            resloc, coloc = self.vh[i].variation.runAllfunctions(
                totoF[ids, :], totoC[ids, :], 0)
            storageF = self.vh[i].variation.runAllrotations(resloc)
            # print(resloc)
            self.F[rangeIdsO[ids], :] = storageF
            self.C[rangeIdsO[ids], :] = coloc

        # print(f"self.F = {self.F}")
        self.C[rangeIdsO, :] /= 2

    def run(self):
        for i in np.arange(self.fractal_parameters.burn):
            self.run1iter(0, True)
        for i in np.arange(1,self.fractal_parameters.niter -1):
            self.run1iter(i, False)
        self.F = self.F * self.fractal_parameters.zoom


    def toImage(self,
                sizeImage=1000,
                coef_forget=1.,
                coef_intensity=.25,
                optional_kernel_filtering=True,
                verbose=0):

        imgtemp = Image.new('RGB', (sizeImage, sizeImage), "black")
        bitmap = np.array(imgtemp).astype(float)
        intensity = np.zeros((sizeImage, sizeImage))

        ## applying offset at the Fractal level
        self.F[:, 0] += self.fractal_parameters.dx
        self.F[:, 1] += self.fractal_parameters.dy

        # ## rotating the Fractal
        self.F = rotation(1,
                          self.fractal_parameters.final_rot,
                          self.F,
                          np.random.uniform(size=self.F.shape[0]))


        F_loc = (sizeImage * (self.F + 1) / 2).astype("i2")

        conditions = np.zeros((F_loc.shape[0], 4), dtype='bool')
        conditions[:, 0] = F_loc[:, 0] < sizeImage
        conditions[:, 1] = F_loc[:, 0] > 0
        conditions[:, 2] = F_loc[:, 1] < sizeImage
        conditions[:, 3] = F_loc[:, 1] > 0

        goods = np.where(np.all(conditions, 1))[0]

        # goods = goods[np.where((F_loc[goods, 0] != sizeImage / 2) | (F_loc[goods, 1] != sizeImage / 2))[0]]


        if verbose > 0:
            print("    number of points in the image: " + str(len(goods)))

        bitmap, intensity = renderImage(
            F_loc, self.C, bitmap, intensity, goods, coef_forget)

        nmax = np.amax(intensity)
        inmax = np.where(intensity == nmax)
        if verbose > 0:
            print("    nmax: " + str(int(nmax)))
            print("    inmax: " + str(inmax))
        intensity = np.power(np.log(intensity + 1) / np.log(nmax + 1), coef_intensity)

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
    # make_serp()
    # make_mess()
    make_final()
    # make_quizz()
