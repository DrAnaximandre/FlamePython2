import class_fractale
import numpy as np


class Movie:

    def __init__(self):
        self.listFractale = []
        self.lock = False

    def addFractale(self, F):
        # todo : check if F is built and locked.
        self.listFractale.append(F)

    def build(self):
        pass    

    def run(self):
        pass    

    def renderGif(self):
        pass


if __name__ == '__main__':

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, .5, -.5, 0, 1])
    a3 = np.array([0, 1, 0, 1, -.8, 1])
    
    burn = 20
    niter = 50
    zoom = .9
    N = 10000

    F1 = Fractale(burn, niter, zoom)
    v1 = Variation()
    v1.addFunction([.4], a1, [linear], .2, [255, 0, 0])
    v1.addFunction([.5], a2, [linear], .2, [0, 255, 0])
    v1.addFunction([.5], a3, [linear], .2, [0, 0, 255])
    F1.addVariation(v1, N)
    F1.build()


    F2 = Fractale(burn, niter, zoom)
    v2 = Variation()
    v2.addFunction([.4], a1, [swirl], .2, [255, 0, 0])
    v2.addFunction([.5], a2, [linear], .2, [0, 255, 0])
    v2.addFunction([.5], a3, [linear], .2, [0, 0, 255])
    F2.addVariation(v1, N)
    F2.build()

    M = Movie()