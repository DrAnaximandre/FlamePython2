import numpy as np
from Fractal import Fractal
from Additives import linear, bubble, swirl,pdj
import matplotlib.pyplot as plt
from quizz import quizz
from Variation import Variation


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
        self.clip = 0.1
        self.W = 3
        self.imsize = 1024
        self.colors = [[250, 0, 0],
                       [0, 250, 0],
                       [0, 0, 250]]
        A = np.array(np.random.uniform(-1.2, 1.2, (self.W, 6)))
        mask_clip = np.abs(A) < self.clip
        not_mask_clip = np.invert(mask_clip)
        A[mask_clip] = 0
        A[not_mask_clip] = A[not_mask_clip]
        self.A = A


def make_quizz(name="key-book-swirl"):

    main_param = ImageParameters(name)

    end = False
    iteration = 0
    while not end:
        iteration += 1
        F1 = Fractal(main_param.burn, main_param.niter, main_param.zoom)
        v1 = Variation(main_param.N)
        for i in range(main_param.W):
            v1.addFunction([.5, 0.2], main_param.A[i, :], [
                           linear, swirl], 0.2, main_param.colors[i % 3])

        F1.addVariation(v1)
        F1.build()
        print("Running")
        F1.runAll()
        print("Generating the image")

        out, bitmap = F1.toImage(main_param.imsize,
                                 coef_forget=main_param.fi,
                                 coef_intensity=main_param.ci,
                                 optional_kernel_filtering=False)

        plt.imshow(bitmap, interpolation='None')
        plt.show()

        main_param, end = quizz(main_param, iteration, out)


def make_final():
    burn = 20
    niter = 50
    zoom = 1
    N = 15000

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, 0, 0, 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])

    F1 = Fractal(burn, niter, zoom)

    v1 = Variation(N)
    v1.addFunction([.5], a1, [linear], .2, [255, 0, 0])
    v1.addFunction([-0.5], a2, [linear], .2, [0, 255, 0])
    v1.addFunction([.5], a3, [linear], .2, [0, 0, 255])
    v1.addFinal([1.5], [-0.5, 0.0003, 1, 0.5, 1, -0.005], [pdj])
    # v1.addRotation((35, np.pi / 4, 0.95))
    F1.addVariation(v1)
    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1024, coef_forget=0.9,
        optional_kernel_filtering=False)
    out.save("final.png")


def make_serp():
    print("init serp triangle")
    burn = 20
    niter = 50
    zoom = 1
    N = 5000

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, 0, 0, 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])

    F1 = Fractal(burn, niter, zoom)

    v1 = Variation(N)
    v1.addFunction([.5], a1, [linear], .2, [255, 0, 0])
    v1.addFunction([.5], a2, [linear], .2, [0, 255, 0])
    v1.addFunction([.5], a3, [linear], .2, [0, 0, 255])

    F1.addVariation(v1)
    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(
        600, coef_forget=.1, optional_kernel_filtering=False)
    out.save("serp.png")


def make_mess():
    print("init mess")
    burn = 20
    niter = 50
    zoom = .45
    N = 5000
    colors = [[70, 119, 125],
              [96, 20, 220],
              [0, 0, 150],
              [82, 171, 165],
              [28, 43, 161]]

    NFunc = 30
    a = np.zeros((NFunc, 6))
    for i in range(NFunc):
        a[i, [((i + 1) * 2) % 6,
              (i * 3 + 2) % 6,
              (i * 4 + 3) % 6]] = 1

    F1 = Fractal(burn, niter, zoom)

    v1 = Variation(N)
    for i in range(NFunc):
        v1.addFunction([.8**(i + 1), (i + 1) * .2], a[i],
                       [linear, bubble], .2, colors[i % 5])

    v1.addRotation((8, np.pi / 4, 1))

    F1.addVariation(v1)

    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(600,
                             coef_forget=.1,
                             coef_intensity=.02,
                             optional_kernel_filtering=True)
    out.save("mess.png")


if __name__ == '__main__':
    make_final()
