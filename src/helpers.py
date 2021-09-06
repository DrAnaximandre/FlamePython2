import numpy as np
from Fractal import Fractal, FractalParameters
from Additives import linear, bubble, swirl,pdj, spherical, sinmoche
import matplotlib.pyplot as plt
from quizz import quizz
from Variation import Variation, VariationParameters


import PIL.ImageOps as pops


class ImageParameters(object):

    def __init__(self, name):

        self.name = name

        self.fractal_parameters = FractalParameters()
        self.variation_parameters = VariationParameters()

        # Parameters of the rendering
        self.ci = 1.0
        self.fi = 0.05
        self.clip = 0.7


        # Parameters of the additives
        self.W = 4
        self.imsize = 1024
        self.colors = [[250, 0, 0],
                       [0, 250, 0],
                       [0, 0, 250]]
        A = np.random.uniform(-1.2, 1.2, (self.W, 6))
        mask_clip = np.abs(A) < self.clip
        A[mask_clip] = 0
        self.A = A
        ws = np.random.uniform(-0.99, 0.99 , size=(self.W,4))
        mask_clip = np.abs(ws) < self.clip
        ws[mask_clip] = 0
        self.ws = ws
        self.p = np.ones(shape=(self.W))



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
                           [linear, swirl, bubble, spherical],
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

        # plt.imshow(bitmap, interpolation='None')
        # plt.show()

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


def make_gray_serp():
    print("init thingy")

    fp = FractalParameters()
    vp = VariationParameters(1000000)

    a1 = np.array([0, 1, 0, -0.5, 0, 1.4])
    a2 = np.array([1, 1, -1, 0, 0, 1])
    a3 = np.array([0, 1, 0, 0.6, 0, 1])

    F1 = Fractal(fp)

    v1 = Variation(vp)
    v1.addFunction([.5], a1, [swirl], .5, [50, 150, 120])
    v1.addFunction([.5], a2, [linear], .5, [90, 190, 255])
    v1.addFunction([.5, -1], a3, [bubble, linear], .5, [200, 80, 40])
    v1.addFinal([1, -0.5, 0.05], [-0.4, 1, 1, -0.5, 1.3, 1e-2], [bubble, bubble, bubble ])

    F1.addVariation(v1)


    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(
        3200, coef_forget=.1, optional_kernel_filtering=False)
    out.save("bigger-thingy2.png")


def make_serp(save=True):
    print("init serp triangle")
    burn = 20
    niter = 50
    zoom = 1.5
    N = 10000

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, 0, 0, 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])

    v1 = Variation(N)
    v1.addFunction([.5], a1, [linear], .5, [255, 0, 0])
    v1.addFunction([.5], a2, [linear], .25, [18, 200, 68])
    v1.addFunction([.5], a3, [linear], .25, [0, 0, 255])


    F1P = FractalParameters(burn, niter, zoom)
    F1 = Fractal(F1P, [v1])

    
    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        600, coef_forget=.1, optional_kernel_filtering=False)
    if save:
        out.save("serp.png")


def big_thingy():



    # colors = np.array([[3, 4, 94],
    # [200,0,0],
    # [0,119,182],
    # [0,150,200],
    # [0,180,216],
    # [72,202,228],
    # [173, 232, 244]])

    colors = np.array([[229,219,255],
    (215,203,255),
    (208,179,255),
    (186,151,255),
    (167,112,255)])


    fp = FractalParameters()
    vp = VariationParameters(1000000)


    A = np.random.choice([0,.25,.5,-.5,-.75,1], size=(5,6),
        p = [0.35,0.1,0.2,0.2,0.05,0.1])

    print(A)
    
    ws = np.random.choice([0,.25,.5,-.5,-.75,1], size=(5,3),
        p = [0.1,0.2,0.35,0.2,0.05,0.1])

    F1 = Fractal(fp)

    v1 = Variation(vp)
    v1.addFunction(ws[0], A[0, :], [bubble, sinmoche, linear], .5, colors[0])
    v1.addFunction(ws[1], A[1,:], [bubble, linear, sinmoche], .5, colors[1])
    v1.addFunction(ws[2], A[2,:], [swirl, linear, spherical], .5, colors[2])
    v1.addFunction(ws[3], A[3,:], [sinmoche, swirl, linear], .5, colors[3])
    v1.addFunction(ws[4], A[4,:],  [sinmoche, swirl, pdj, ], .5, colors[4])
    v1.addFinal([0.5, 0.75], [1, 0.5, 1, -1, 1, 1], [swirl, bubble ])

    v1.addRotation((3, 2 * np.pi / 3, 1))
    F1.addVariation(v1)


    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(
        2000, coef_forget=.15, optional_kernel_filtering=True)
    out.save("wb2.png")

    pops.invert(out).save("bob.png")


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
    make_serp()
