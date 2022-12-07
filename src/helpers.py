from functools import partial
import numpy as np
from Fractal import Fractal, FractalParameters
from Additives import linear, bubble, swirl,pdj, spherical, sinmoche
from quizz import quizz
from Variation import Variation, VariationParameters
from typing import Tuple

import PIL.ImageOps as pops

from joblib import Parallel, delayed

import glob
from natsort import natsorted
from moviepy.editor import *



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


def morning_fractal(i=0,save=True):
    print("Morning fractal")
    burn = 20
    niter = 50
    zoom = 1
    N = 100000

    a1 = np.array([-0.105+0.6, 1, -0.1, 0.2, 0, 0.7+0.7*1.05])
    a2 = np.array([1, 0.5+0.7, -0.1, 0.2, 0.2, -2+0.6])
    a3 = np.array([-1+0.005, 1, -0.1, -2+0.4*3+0.05, 0, 1])

    v1 = Variation(N)
    v1.addFunction([.5,0.25], a1, [linear, swirl], .25, [255, 0, 0])
    v1.addFunction([.5,(1-1.2+0.9/10)**(i+1)], a2, [linear,bubble], .25, [218,165,32])
    v1.addFunction([.5, -(1-0.9)/10-0.5-0.1+0.6/10], a3, [linear,swirl], .25, [176, 48, 96])

    F1P = FractalParameters(burn, niter, zoom)
    F1 = Fractal(F1P, [v1])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"morning_fractal_finalii_1{i}.png")

def aftervening_fractal(i=0,save=True):
    print("Aftervening fractal")
    burn = 20
    niter = 50
    zoom = 1
    N = 100000

    a1 = np.array([1,0.2,1,-1,0.5,-0.5])
    a2 = np.array([-1,1,1,-1.5,1,1])
    a3 = np.array([0.2, 1.5, -0.1, -0.75, 0.07, 1])
    a4 = np.array([0.5,0.5-0.2,2*np.sin(i*np.pi*2),0.5,-0.5,1])

    v1 = Variation(N)
    v1.addFunction([.15], a1, [linear], .25, tuple(int("E2F20D"[j:j+2], 16) for j in (0, 2, 4)))
    v1.addFunction([.25,0.6], a2, [linear,bubble], .25, tuple(int("0DF290"[j:j+2], 16) for j in (0, 2, 4)))
    v1.addFunction([.5*0.2,1], a3, [linear, spherical], .25, tuple(int("1D0DF2"[j:j+2], 16) for j in (0, 2, 4)))
    v1.addFunction([.8+3,2], a4, [swirl, linear], .25*i, tuple(int("F20D6F"[j:j + 2], 16) for j in (0, 2, 4)))
    v1.addRotation((12,i*2*np.pi,0.75))

    v1.addFinal([0.02
                    ,0.6],(a1+a2+a3+a4)*np.sin(i*np.pi*2)/2+0.1,[spherical, linear])

    F1P = FractalParameters(burn, niter, zoom)
    F1 = Fractal(F1P, [v1])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"aftervening_movie_u{i*400}.png")


def get_RGB_from_hex(hex: str) -> Tuple[int]:
    return tuple(int(hex[j:j + 2], 16) for j in (0, 2, 4))

def swirl_second(i=0,save=True):
    print("Test dx dy")
    burn = 20
    niter = 50
    zoom = 1
    N = 25000

    a1 = np.array([0, 1, 0, 0, 0, 1])
    a2 = np.array([1, 1, 0,  0, 0, 1])
    a3 = np.array([0, 1,  0, 1, 0, 1])
    a5 = np.array([-1, 1,  0, 0, 0, 1])
    a8 = np.array([-1, 1,  0, 0, 0, 1])
    a9 = np.array([0, 1, 0, -1, 0, 1])

    v1 = Variation(N)
    v1.addFunction([.5, 0.25*np.sin(i*np.pi*2)], a1, [linear, swirl], .25, [210,10,60])
    v1.addFunction([.5], a2, [linear], .25, get_RGB_from_hex("DDAA22"))
    v1.addFunction([.5,0.25*(1-np.cos(i*np.pi*2))], a3, [linear, swirl], .25, get_RGB_from_hex("22DDAA"))

    v2 = Variation(N)
    v2.addFunction([.5], a1, [linear], .25, [210, 10, 60])
    v2.addFunction([.5, 0.25*(1-np.cos(i*np.pi*4))], a5, [linear, swirl], .25, get_RGB_from_hex("DDAA22"))
    v2.addFunction([.5, 0.5*np.sin(i*np.pi*2)], a3, [linear, swirl], .25, get_RGB_from_hex("22DDAA"))

    v3 = Variation(N)
    v3.addFunction([.5, 0.5*(1-np.cos(i*np.pi*2))], a1, [linear, swirl], .25, [210, 10, 60])
    v3.addFunction([.5], a8, [linear], .25, get_RGB_from_hex("DDAA22"))
    v3.addFunction([.5, 0.25*np.sin(i*np.pi*2)], a9, [linear, swirl], .25, get_RGB_from_hex("22DDAA"))

    v4 = Variation(N)
    v4.addFunction([.5,0.5*np.sin(i*np.pi*2)], a1, [linear, swirl], .25, [210, 10, 60])
    v4.addFunction([.5], a2, [linear], .25, get_RGB_from_hex("DDAA22"))
    v4.addFunction([.5, 0.25*(1-np.cos(i*np.pi*4))], a9, [linear, swirl], .25, get_RGB_from_hex("22DDAA"))

    F1P = FractalParameters(burn, niter, zoom, 0.0, 0.0, -np.pi*2*i)
    F1 = Fractal(F1P, [v1, v2, v3, v4])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"second_test_swirl{i*400}.png")


def bubble_again(i=0,save=True):
    name = "bubble-roar-second"

    burn = 20
    niter = 50
    zoom = 1
    N = 25000

    a1 = np.array([0, 1, 0.25*(1-np.cos(i*np.pi*4)), 0, 0, 1])
    a2 = np.array([1, 1, 0,  0.25*(1-np.cos(i*np.pi*8)), 0, 1])
    a3 = np.array([0, 1,  0, 1, 0, 1])
    a5 = np.array([-1, 1,  0, 0, 0, 1])
    a8 = np.array([-1, 1,  0, 0, 0, 1])
    a9 = np.array([0, 1, 0.25*(1-np.cos(i*np.pi*4)), -1, 0, 1])

    z_ = (1+np.cos(i*np.pi*4))/2
    zf_ = (1 + np.cos(i * np.pi * 2)) / 2
    o_ = (1 + np.sin(i * np.pi * 2)) / 2


    v1 = Variation(N)
    v1.addFunction([.5, 0.25*np.sin(i*np.pi*2)], a1, [linear, bubble], .25, [152,78,255*z_])
    v1.addFunction([.5], a2, [linear], .25, [240,78,255])
    v1.addFunction([.5,0.25*(1-np.cos(i*np.pi*2))], a3, [linear, bubble], .25, [78,108,255])

    v2 = Variation(N)
    v2.addFunction([.5], a1, [linear], .25, [152,78,255])
    v2.addFunction([.5, 0.25*(1-np.cos(i*np.pi*4))], a5, [linear, bubble], .25, [240,78,255])
    v2.addFunction([.5, 0.5*np.sin(i*np.pi*2)], a3, [linear, bubble], .25, [78,108*zf_ ,220])

    v3 = Variation(N)
    v3.addFunction([.5, 0.5*(1-np.cos(i*np.pi*2))], a1, [linear, spherical], .25, [152,78*z_,255])
    v3.addFunction([.5], a8, [linear], .25, [240,78,255*zf_])
    v3.addFunction([.5, 0.25*np.sin(i*np.pi*2)], a9, [linear, spherical], .25, [78*z_ ,108*2*o_,200])

    v4 = Variation(N)
    v4.addFunction([.5,0.5*np.sin(i*np.pi*2)], a1, [linear, bubble], .25, [152,78,255*zf_])
    v4.addFunction([.5], a2, [linear], .25, [240,78*z_*z_,255])
    v4.addFunction([.5, 0.25*(1-np.cos(i*np.pi*4))], a9, [linear, bubble], .25, [78,108,210])

    F1P = FractalParameters(burn, niter, zoom, 0.0, 0.0, 2*np.pi*2*i)
    F1 = Fractal(F1P, [v1, v2, v3, v4])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{name}-{i*500}.png")

def mixedbag(i=0, save=True):
    name = "mixedbag"

    burn = 20
    niter = 50
    zoom = 1
    N = 25000

    a1 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), 0, 0, 1])
    a2 = np.array([1, 1, 0, 0.25 * (1 - np.cos(i * np.pi * 8)), 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])
    a5 = np.array([-1, 1, 0, 0, 0, 1])
    a8 = np.array([-1, 1, 0, 0, 0, 1])
    a9 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), -1, 0, 1])

    z_ = (1 + np.cos(i * np.pi * 4)) / 2
    zf_ = (1 + np.cos(i * np.pi * 2)) / 2
    o_ = (1 + np.sin(i * np.pi * 2)) / 2

    c1 = [c*1.5 for c in [39, 25, 130]]
    c2 = [c*1.5 for c in [128, 115, 43]]
    c3 = [c*1.5 for c in [128, 73, 43]]

    v1 = Variation(N)
    v1.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], -a1, [linear, bubble], .25, [c*z_ for c in c1])
    v1.addFunction([.5], -a2, [linear], .25, c2)
    v1.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 2))], -a3, [linear, swirl], .25, c3)

    v2 = Variation(N)
    v2.addFunction([.5], -a1, [linear], .25, c2)
    v2.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 4))], -a5, [linear, bubble], .25, c1)
    v2.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], -a3, [linear, spherical], .25, [c3[i]*z_ if i ==0 else c3[i] for i in range(3)])

    v3 = Variation(N)
    v3.addFunction([.5, 0.5 * (1 - np.cos(i * np.pi * 2))], -a1, [linear, spherical], .25, [c2[i]*zf_ if i==1 else c2[i] for i in range(3)])
    v3.addFunction([.5], -a8, [linear], .25,  [c2[i]*o_ if i==2 else c2[i] for i in range(3)])
    v3.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], -a9, [linear, swirl], .25, c1)

    v4 = Variation(N)
    v4.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], -a1, [linear, swirl], .25, [c*z_ for c in c3])
    v4.addFunction([.5], -a2, [linear], .25,  [c1[i]*o_ if i==0 else c1[i] for i in range(3)])
    v4.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 4))], -a9, [linear, bubble], .25, c2)

    F1P = FractalParameters(burn, niter, zoom, 0.0, 0.0, 2 * np.pi * 2 * i)
    F1 = Fractal(F1P, [v1, v2, v3, v4])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{name}-{i * 1000}.png")



def contrary_motion(i=0, save=True):
    name = "cm"

    burn = 20
    niter = 50
    zoom = 1
    N = 25000

    a1 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), 0, 0, 1])
    a2 = np.array([1, 1, 0, 0.25 * (1 - np.cos(i * np.pi * 8)), 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])
    a5 = np.array([-1, 1, 0, 0, 0, 1])
    a8 = np.array([-1, 1, 0, 0, 0, 1])
    a9 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), -1, 0, 1])

    z_ = (1 + np.cos(i * np.pi * 4)) / 2
    zf_ = (1 + np.cos(i * np.pi * 2)) / 2
    o_ = (1 + np.sin(i * np.pi * 2)) / 2

    c1 =  [91, 206, 250]
    c2 = [245, 169, 184]
    c3 =  [255, 255, 255]

    v1 = Variation(N)
    v1.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], -a1, [linear, bubble], .25, [c*z_ for c in c1])
    v1.addFunction([.5], -a2, [linear], .25, c2)
    v1.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 2))], -a3, [linear, swirl], .25, c3)
    v1.addRotation((1,np.pi * 2 * i,1))

    v2 = Variation(N)
    v2.addFunction([.5], -a1, [linear], .25, c2)
    v2.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 4))], -a5, [linear, bubble], .25, c1)
    v2.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], -a3, [linear, spherical], .25, [c3[i]*z_ if i ==0 else c3[i] for i in range(3)])
    v2.addRotation((1, -np.pi * 2 * i, 1))


    v3 = Variation(N)
    v3.addFunction([.5, 0.5 * (1 - np.cos(i * np.pi * 2))], -a1, [linear, spherical], .25, [c2[i]*zf_ if i==1 else c2[i] for i in range(3)])
    v3.addFunction([.5], -a8, [linear], .25,  [c2[i]*o_ if i==2 else c2[i] for i in range(3)])
    v3.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], -a9, [linear, swirl], .25, c1)
    v3.addRotation((1,np.pi * 4 * i,1))


    v4 = Variation(N)
    v4.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], -a1, [linear, swirl], .25, [c*z_ for c in c3])
    v4.addFunction([.5], -a2, [linear], .25,  [c1[i]*o_ if i==0 else c1[i] for i in range(3)])
    v4.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 4))], -a9, [linear, bubble], .25, c2)
    v4.addRotation((1, -np.pi * 4 * i, 1))

    F1P = FractalParameters(burn, niter, zoom, 0.0, 0.0, 2 * np.pi * 2 * i)
    F1 = Fractal(F1P, [v1, v2, v3, v4])

    print("Running")
    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.6,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{name}-{i * 1000}.png")


def short_restart(i=0, name="short_restart", save=True):

    folder_name = f"../images/{name}/"
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    burn = 20
    niter = 50
    zoom = 1
    N = 2500

    a1 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), 0, 0, 1])
    a2 = np.array([1, 1, 0, 0.25 * (1 - np.cos(i * np.pi * 8)), 0, 1])
    a3 = np.array([0, 1, 0, 1, 0, 1])
    a5 = np.array([-1, 1, 0, 0, 0, 1])
    a8 = np.array([-1, 1, 0, 0, 0, 1])
    a9 = np.array([0, 1, 0.25 * (1 - np.cos(i * np.pi * 4)), -1, 0, 1])

    z_ = (1 + np.cos(i * np.pi * 4)) / 2
    zf_ = (1 + np.cos(i * np.pi * 2)) / 2 - 1
    o_ = (1 + np.sin(i * np.pi * 2)) / 2

    c1 = [209, 34, 41]
    c2 = [246, 138, 30]
    c3 = [253, 224, 26]
    c4 = [0, 121*1.2, 64*1.2]
    c5 = [36*1.2, 64*1.2, 142*1.2]
    c6 = [115*1.2, 41*1.2, 130*1.2]



    v1 = Variation(N)
    v1.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], a1, [linear, swirl], .25, c1)
    v1.addFunction([.5], a2, [linear], .25, c2)
    v1.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 2))], a3, [linear, swirl], .25,  c3)
    v1.addRotation((1,np.pi * 2 * i, 1))

    v2 = Variation(N)
    v2.addFunction([o_], a1, [linear], .25, c4)
    v2.addFunction([.5, 0.25 * (1 - np.cos(i * np.pi * 4))], a5, [linear, swirl], .25, c5)
    v2.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], a3, [linear, swirl], .25, c6)
    v2.addRotation((1, -np.pi * 2 * i, 1))

    v3 = Variation(N)
    v3.addFunction([.5, 0.5 * (1 - np.cos(i * np.pi * 2))], a1, [linear, swirl], .25, c6)
    v3.addFunction([.5, zf_], a8, [linear, bubble], .25,  c2)
    v3.addFunction([.5, 0.25 * np.sin(i * np.pi * 2)], a9, [linear, swirl], .25, c3)


    v4 = Variation(N)
    v4.addFunction([.5, 0.5 * np.sin(i * np.pi * 2)], a1, [linear, swirl], .25, c4)
    v4.addFunction([0.5], a2, [linear], .25,  c5)
    v4.addFunction([.5, 0*(o_-0.5)], a9, [linear, swirl], .25, c1)
    v4.addRotation((1, np.pi * 4 * i, 1))

    F1P = FractalParameters(burn, niter, zoom, 0.0, 0.0,0)
    F1 = Fractal(F1P, [v1, v2, v3, v4])

    F1.run()
    out, bitmap = F1.toImage(
        600
        ,
        coef_forget=0.3,
        coef_intensity=0.8,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{folder_name}{name}-{i * 250}.png")

def uff(i=0, name="uff", save=True):

    folder_name = f"../images/{name}/"
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    burn = 20
    niter = 50
    N = 25000


    z_ = (1 + np.cos(i * np.pi * 4)) / 2 # commence à 1
    zf_ = (1 + np.cos(i * np.pi * 8)) / 2 # commence à 1
    o_ = (1 + np.sin(i * np.pi * 16)) / 2 # commence à 05
    of_ = (1 + np.cos(i * np.pi * 4 + 2*np.pi/2)) / 2 # commence à 0

    a1 = np.array([1, z_, 0, 1, 0, 1])
    a2 = np.array([z_, 1, 0, of_, 0, 1])
    a3 = np.array([of_, 1, 0, 1, o_, 1])

    c1 = [255, 255, 255]
    c2 = [200, 200, 200]
    c3 = [225, 225, 225]

    v1 = Variation(N)
    v1.addFunction([.5, of_*2], a1, [linear, swirl], .25, c1)
    v1.addFunction([.5], a2, [linear], .25, c2)
    v1.addFunction([.5, of_], a3, [linear, swirl], .25,  c3)
    v1.addFunction([ o_, of_], a3+a2, [swirl, bubble], of_, c3)
    v1.addRotation(90)
    v1.addFinal([0.4,0.1*zf_],(a1+a2+a3)/3,[linear, spherical])

    F1P = FractalParameters(burn, niter, 1
                            , 0.0, 0.0, 0)
    F1 = Fractal(F1P, [v1])

    F1.run()
    print("Generating the image")
    out, bitmap = F1.toImage(
        1280,
        coef_forget=0.3,
        coef_intensity=0.8,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{folder_name}{name}-{i * 250}.png")



if __name__ == '__main__':

    fps = 25
    n_im = 250
    name = "demo"

    Parallel(n_jobs=-2)(
        delayed(partial(short_restart, name=name))(
            (i)/n_im) for i in range(n_im+1)
    )

    base_dir = os.path.realpath(f"../images/{name}/")
    file_list = glob.glob(f'{base_dir}/{name}*.png')
    file_list_sorted = natsorted(file_list, reverse=False)

    clips = [ImageClip(m).set_duration(1/fps)
             for m in file_list_sorted]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(f"{base_dir}/{name}.mp4", fps=fps)
