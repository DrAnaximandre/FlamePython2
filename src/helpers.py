import numpy as np
from Fractal import Fractal, FractalParameters
from Additives import linear, bubble, swirl,pdj
import matplotlib.pyplot as plt
from quizz import quizz
from Variation import Variation, VariationParameters





class ImageParameters(object):

    def __init__(self, name):

        self.name = name

        self.fractal_parameters = FractalParameters()
        self.variation_parameters = VariationParameters()

        # Parameters of the rendering
        self.ci = 0.5
        self.fi = 0.05
        self.clip = 0.1


        # Parameters of the additives
        self.W = 5
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
        self.ws = np.random.uniform(size=(self.W,2))
        self.p = np.random.uniform(size=(self.W))



def make_quizz(name="key-book-swirl"):

    main_param = ImageParameters(name)

    end = False
    iteration = 0
    while not end:
        iteration += 1
        F1 = Fractal(main_param.fractal_parameters)
        v1 = Variation(main_param.variation_parameters)
        for i in range(main_param.W):
            v1.addFunction(main_param.ws[i],
                           main_param.A[i, :],
                           [linear, swirl],
                           main_param.p[i],
                           main_param.colors[i % 3])

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

    v1.addFinal([0.95], [-0.5, 0.0003, 1, 0.5, 1, -0.005], [linear])
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
    v1.addFunction([.5], a1, [linear], .5, [255, 0, 0])
    v1.addFunction([.5], a2, [linear], .25, [0, 255, 0])
    v1.addFunction([.5], a3, [linear], .25, [0, 0, 255])

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
    N = 15000
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
    make_quizz("wow-another-one")
