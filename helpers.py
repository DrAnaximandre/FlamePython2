import numpy as np
from class_fractale import Fractale, Variation
from utils import linear

def make_serp():
    print("init serp triangle")
    burn = 20
    niter = 50
    zoom = 1
    N = 10000

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, 0, 0, 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])

    F1 = Fractale(burn, niter, zoom)

    v1 = Variation()
    v1.addFunction([.5], a1, [linear], .2, [255, 0, 0])
    v1.addFunction([.5], a2, [linear], .2, [0, 255, 0])
    v1.addFunction([.5], a3, [linear], .2, [0, 0, 255])

    F1.addVariation(v1, N)
    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out, bitmap = F1.toImage(600, coef_forget=.1, optional_kernel_filtering=False)
    out.save("serp.png")

def make_mess():
    print("init mess")
    burn = 50
    niter = 100
    zoom = .45
    N = 20000
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

    F1 = Fractale(burn, niter, zoom)

    v1 = Variation()
    for i in range(NFunc):
        v1.addFunction([.8**(i + 1), (i + 1) * .2], a[i],
                       [linear, bubble], .2, colors[i % 5])

    v1.addRotation((8, np.pi / 4, 1))

    F1.addVariation(v1, N)

    F1.build()
    print("Running")
    F1.runAll()
    print("Generating the image")
    out = F1.toImage(600,
                     coef_forget=.1,
                     coef_intensity=.02,
                     optional_kernel_filtering=True)
    out.save("mess.png")