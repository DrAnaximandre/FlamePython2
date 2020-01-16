from utils import *
import numpy as np
from PIL import Image, ImageFilter
import sys
import matplotlib.pyplot as plt
from quizz import quizz
from Function import Function


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

        bitmap, intensity = renderImage(F_loc, self.C, bitmap,
                                        intensity, goods, coef_forget)

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


class ImageParameters(object):

    def __init__(self, name):
        self.name = name
        self.burn = 5
        self.niter = 25
        self.zoom = 1
        self.N = 5000
        self.end = False
        self.ci = 0.5
        self.fi = 0.05
        self.clip=0.1
        self.W = 3
        self.imsize = 1024
        self.colors = [[250, 0, 0],
                  [0, 250, 0],
                  [0, 0, 250]]
        A = np.array(np.random.uniform(-1.2,1.2, (self.W,6)))
        mask_clip = np.abs(A)<self.clip
        not_mask_clip = np.invert(mask_clip)
        A[mask_clip] = 0
        A[not_mask_clip] = A[not_mask_clip]
        self.A = A


if __name__ == '__main__':

    main_param = ImageParameters("key-book-swirl")

    end = False
    iteration =0
    while not end:
        iteration +=1
        F1 = Fractale(main_param.burn, main_param.niter, main_param.zoom)
        v1 = Variation()
        for i in range(main_param.W):
            v1.addFunction([.5,0.2], main_param.A[i,:], [linear, swirl], 0.2, main_param.colors[i%3])

        F1.addVariation(v1, main_param.N)
        F1.build()
        print("Running")
        F1.runAll()
        print("Generating the image")


        out, bitmap = F1.toImage(main_param.imsize, 
            coef_forget=main_param.fi, 
            coef_intensity=main_param.ci,
            optional_kernel_filtering=False)
        
        plt.imshow(bitmap, interpolation = 'None')
        plt.show()

        main_param, end = quizz(main_param,iteration, out)

    # from helpers import make_serp
    # make_serp()
    # make_mess()
