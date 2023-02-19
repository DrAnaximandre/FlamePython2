import os
from functools import partial

import numpy as np

from src.Additives import swirl, linear, bubble, spherical, pdj
from src.Fractal import FractalParameters, Fractal
from src.Function import Function
from src.Variation import Variation



cr1 = [255, 0, 0]
cr2 = [255, 64, 64]
cr3 = [255, 128, 128]
cr4 = [255, 192, 192]
cr5 = [255, 255, 255]
cr6 = [192, 0, 0]

cb1 = [0, 0, 255]
cb2 = [64, 64, 255]
cb3 = [128, 128, 255]
cb4 = [192, 192, 255]
cb5 = [255, 255, 255]
cb6 = [0, 0, 192]

cg1 = [0, 255, 0]
cg2 = [0, 200, 0]
cg3 = [10, 165, 20]


R = [cr1, cr2, cr3, cr4, cr5, cr6]
B = [cb1, cb2, cb3, cb4, cb5, cb6]
G = [cg1, cg2, cg3]

def ImageMaker( #i=0,  # frame number
                r_i=0, # ratio of the frame number over the max number of frames
                name="purple",
                save = True):

    folder_name = f"../images/{name}/"
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #adapt later
    i = int(r_i * 50)

    alpha = (np.cos(2 * np.pi * r_i) + 1) / 2
    beta = (np.cos(r_i * np.pi * 4 * alpha + np.pi/2)*10 + 11) / 21

    burn = 10
    niter = 30
    zoom = 1
    N = 50000

    M = 151

    A = np.zeros((M, 6))
    for j in range(M):
        A[j, j%6] = 10/(1+j)
    for j in range(M):
        for k in range(6):
            A[j, k] += np.cos(np.sqrt(k) + 4 * np.pi * j / M) * 0.456

    for j in range(M):
        A[j, (j+1)%6] = 0.76* np.cos(j+r_i * np.pi * 2)
        A[j, (j*j+1)%6] = 0.77* np.sin(1+ j-r_i * np.pi * 2)
        A[-j, (j*j)%6] = - 0.554 * np.cos(np.sqrt(j+1) + r_i * np.pi * 2)
        A[-j, (-j)%6] = - np.sin(np.sqrt(j+1) - r_i * np.pi * 4)


    v1 = Variation(N)

    for j in range(M):
        index_c1 = j % len(B)
        index_c2 = (j * j + j) % len(R)
        index_c3 = (j * j + j + 1) % len(B)
        index_c4 = (j + 2) % len(B)
        beta_j =  (np.cos(j+ r_i * np.pi * 2) + 1) / 2

        c1 = np.array(B[index_c1]) * beta_j + np.array(B[index_c3]) * (1-beta_j)
        c2 = np.array(R[index_c2]) * beta_j + np.array(B[index_c4]) * (1-beta_j)

        color = c1 * alpha + c2 * (1-alpha)
        v1.addFunction([0.6 + 0.4 * np.sin((1+alpha)*(1-r_i)*2*np.pi), (1-beta)*0.005, 0.003, 0.003*beta_j/2],
                       A[(j-1)%M, :], [bubble, swirl, pdj, linear], 1/M,  col=color)

    #v1.addRotation((4, r_i*np.pi*2, 0.88))
    #v1.addFinal([0.5, -.07, 0.01],[0.5, 1.1, alpha, -0.005, -1.1, 0.4321], [bubble, linear, spherical])

    F1P = FractalParameters(burn, niter, zoom, 0, 0, 0)
    F1 = Fractal(F1P, [v1])

    F1.run()
    out, bitmap = F1.toImage(
        1024,
        coef_forget=1.3,
        coef_intensity=1.8,
        optional_kernel_filtering=False)
    if save:
        out.save(f"{folder_name}{name}-{i}.png")


def BleuNoir( #i=0,  # frame number
                r_i=0, # ratio of the frame number over the max number of frames
                name="purple",
                save = True):

    folder_name = f"../images/{name}/"
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #adapt later
    i = int(r_i * 250)
    if i%10 == 0:
        print(i)
    else:
        alpha = (np.cos(2 * np.pi * r_i) + 1) / 2
        beta = (np.cos(r_i * np.pi * 2 + np.pi/2) + 1) / 2

        burn = 10
        niter = 30
        zoom = 1.5
        N = 30000

        M = 151

        A = np.zeros((M, 6))
        for j in range(M):
            A[j, j%6] = 10/(1+j)
        for j in range(M):
            for k in range(6):
                A[j, k] += np.cos(np.sqrt(k) + 4 * np.pi * j / M) * 0.456

        for j in range(M):
            A[j, (j+1)%6] = 0.76* np.cos(j+r_i * np.pi * 2)
            A[j, (j*j+1)%6] = 0.77* np.sin(1+ j-r_i * np.pi * 2)
            A[-j, (j*j)%6] = - 0.554 * np.cos(np.sqrt(j+1) + r_i * np.pi * 2)
            A[-j, (-j)%6] = - np.sin(np.sqrt(j+1) - r_i * np.pi * 4)


        v1 = Variation(N)

        for j in range(M):
            index_c1 = j % len(B)
            index_c2 = (j * j + j) % len(B)
            index_c3 = (j * j + j + 1) % len(G)
            index_c4 = (j + 2) % len(R)
            beta_j =  (np.cos(j+ r_i * np.pi * 2) + 1) / 2

            c1 = np.array(B[index_c1]) * beta_j + np.array(B[index_c3]) * (1-beta_j)
            c2 = np.array(G[index_c2]) * beta_j + np.array(R[index_c4]) * (1-beta_j)

            color = c1 * alpha + c2 * (1-alpha)
            v1.addFunction([0.6*alpha, -0.49*(1-alpha)], A[j, :], [swirl, bubble], 1/M,  col=color)

        v1.addRotation((6, r_i*np.pi*2, 0.88))
        v1.addFinal([0.5, -.07, 0.01],[0.5, 1.1, alpha, -0.005, -1.1, 0.4321], [bubble, linear, spherical])

        F1P = FractalParameters(burn, niter, zoom, 0, 0, 0)
        F1 = Fractal(F1P, [v1])

        F1.run()
        out, bitmap = F1.toImage(
            1024,
            coef_forget=0.3,
            coef_intensity=0.8,
            optional_kernel_filtering=True)
        if save:
            out.save(f"{folder_name}{name}-{i}.png")



from LFOs import *


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
def LinearBlossomReturns(r_i=0,
                         name="LBR"):


    folder_name = f"../images/{name}/"


    create_folder(folder_name)

    # adapt later
    i = int(r_i * 25)

    alpha = sLFO(min = 0.1, max=1)(r_i)
    gamma = sLFO(min=0, max=1, speed=2*np.pi, phase=1)(r_i)


    burn = 10
    niter = 45
    zoom = 1 # sLFO(min = 0.971, max=1, speed=4*np.pi)(sLFO()(r_i))
    N = 50000


    M = 15

    A = np.ones((M, 6)) *0.004
    for t in range(M):
        beta_t = sLFO(min = 0.1, max=1, speed=4*np.pi)(t/M)
        for j in range(6):
            A[t, j] +=  np.where(j>2 == 0, 1, -1.05) * np.cos(np.sqrt(j+alpha) + 2 * np.pi * t / M) * 0.810256
            A[t, j] +=  np.where(j%3 == 0, 1, -1) * sLFO(phase = np.sqrt(j+beta_t), speed=2*np.pi, max = 0.34999)(r_i)
            A[t, j] -=  np.where(j%5 == 0, -0.4, -0.4) * sLFO(min=0.5, max=0.6, phase=t)(r_i) *t/M
            # A[t, j] += np.where((t*j) % 4 == 0, -0.08, 0.09) * sLFO(min=-0.1, max=0.6, phase=t/M)(r_i)
            # A[t, j] += np.where(j % 3 == 0, -0.3, 0.3) * sLFO(min=-1, max= 0.6, phase=t+j)(r_i)*beta_t

    v = Variation(N)
    for t in range(M):
        beta_t = sLFO(min = 0, max=0.1, speed=4*np.pi)(t/(M*2) + r_i/2)

        c1 = R[t % len(R)]
        c2 = B[t**2 % len(B)]
        color = np.array(c1) * gamma + np.array(c2) * (1-gamma)
        color = np.array(c1) * alpha + np.array(color) * (1-alpha)
        v.addFunction([.5, beta_t], A[t,:], [partial(linear, dx=0, dy=0), spherical], 1/M, color)

    F1P = FractalParameters(burn, niter, zoom, 0, 0, 2*np.pi*r_i)
    F1 = Fractal(F1P, [v])
    F1.run()
    out, bitmap = F1.toImage(
        1024,
        coef_forget=1.003,
        coef_intensity=1.8,
        optional_kernel_filtering=False)
    out.save(f"{folder_name}{name}-{i}.png")



