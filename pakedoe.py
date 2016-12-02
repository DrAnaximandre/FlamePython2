from class_fractale import *
import numpy as np
from pyDOE import *

sethue0 = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
           [150, 150, 150], [200, 200, 200]]  # RGB blanc

sethue1 = [[116, 221, 97], [117, 114, 254], [130, 202, 91],
           [90, 69, 184], [38, 115, 9]]  # violet vert

sethue2 = [[1, 205, 178], [254, 119, 249], [247, 160, 5],
           [111, 202, 148], [232, 150, 64]]  # rose ocre vert

sethue3 = [[114, 160, 226], [255, 0, 0], [0, 0, 255], [
    200, 0, 200], [211, 123, 203]]  # violet bleu rouge

sethue4 = [[70, 119, 125],
           [96, 20, 220],
           [0, 0, 150],
           [82, 171, 165],
           [28, 43, 161]]  # bleu

sethue5 = [[255, 0, 255], [176, 75, 215], [112, 10, 207],
           [255, 255, 255], [200, 200, 200]]

sethue6 = [[90, 104, 103],
           [192, 255, 187],
           [21, 255, 0],
           [255, 0, 0], [86, 29, 35]
           ]  # verts et rouge


def gethue():
    nhue = np.random.randint(7)
    hue = eval("sethue" + str(nhue))
    # permutation?
    np.random.shuffle(hue)
    return(hue)


def addV():

    hue = gethue()
    idr = np.random.rand()
    if idr < .5:
        Nf = np.random.random_integers(3, 6)
        Nadd = np.random.random_integers(2, 3)
    elif idr < 1:
        Nf = np.random.random_integers(3, 5)
        Nadd = np.random.random_integers(4, 6)
    # else:
    #     Nf = np.random.random_integers(30, 50)
    #     Nadd = 2
    print(Nf, Nadd)

    ida = np.random.rand()
    if ida < .3:
        A = lhs(n=6, samples=Nf, criterion='center') * \
            2 - .1 + np.random.normal(0, .5, size=(Nf, 6))
        A[np.abs(A) < .2] = 0
        if np.random.rand() < .25:
            A[A > 0] = 1
        if np.random.rand() < .25:
            A[A < 0] = -1
    else:
        A = np.ones((Nf, 6)) + np.random.normal(0, .05, size=(Nf, 6))
        for a in range(int(Nf * 2.5)):
            A[np.random.randint(0, Nf), np.random.randint(0, 6)] = 0
        for a in range(int(Nf * 1.3)):
            A[np.random.randint(0, Nf), np.random.randint(0, 6)] *= -1

    idr = np.random.rand()
    if idr < .5:
        W = lhs(n=Nadd, samples=Nf, criterion='center') * 2 - 1
        W[np.abs(W) < .2] = .2
        # W =  np.random.normal(0, .9, size=(Nf, Nadd))
    else:
        W = np.ones((Nf, Nadd)) * .5 + \
            np.random.normal(0, .1, size=(Nf, Nadd))

        for w in range(int(Nf * 2)):
            W[np.random.randint(0, Nf), np.random.randint(0, Nadd)] *= -1

    ladd0 = [linear]
    if np.random.rand() < .5:
        ladd1 = [sinmoche, expinj, spherical, swirl, bubble]
    else:
        ladd1 = [sinmoche, expinj, spherical, swirl, bubble,
                 Rsinmoche, Rexpinj, Rspherical, Rswirl, Rbubble]

    np.random.shuffle(ladd1)
    for i in range(len(ladd1)):
        if np.random.rand() < .2:
            ladd1.pop()

    v1 = Variation()

    limlin = np.random.uniform(.3, .8)

    for f in range(Nf):

        ladd = []
        if np.random.rand() < .5:
            limadd = np.random.random_integers(2, Nadd)
        else:
            limadd = Nadd
        for a in range(limadd):
            if np.random.rand() < limlin:
                ladd.append(ladd0[np.random.randint(len(ladd0))])
            else:
                ladd.append(ladd1[np.random.randint(len(ladd1))])

        localhue = hue[f % 5] + np.random.random_integers(-50, 50, size=3)
        localhue[localhue < 0] = 0
        localhue[localhue > 255] = 255

        v1.addFunction(W[f, :a], A[f, :], ladd, np.random.rand(), localhue)

    if np.random.rand() < .25:
        laddf = [linear]
        for t in range(np.random.randint(0, 2)):
            laddf.append(ladd1[np.random.randint(len(ladd1))])
        v1.addFinal(np.random.normal(0, .5, len(laddf)),
                    A[np.random.randint(Nf), :],
                    laddf)

    if np.random.rand() < .2:
        for i in range(np.random.randint(5)):
            v1.addRotation((np.random.randint(15), np.random.rand() * 3.14,
                            np.random.uniform(.8, 1.2)))


    return(v1)


if __name__ == '__main__':

    # yo do not touch this
    burn = 20
    niter = 20
    zoom = 1.2
    N = 100000

    # ok now you can play with all the parameters

    for i in range(20):
        print("---- " + str(i) + " ----")
        F1 = Fractale(burn, niter, zoom)
        v1 = addV()

        # nope! cant touch this
        # stop playing with the parameters now
        F1.addVariation(v1, N)

        F1.build()
        F1.runAll()
        rescore = F1.toScore()
        f = open("doe/doe.txt", 'a')
        f.write(str(i) + ";" + rescore + "\n")

        print("Generating the image")
        out = F1.toImage(1000, coef_forget=.05,
                         coef_intensity=.22,
                         optional_kernel_filtering=False)
        out.save("doe/doe" + str(i) + ".png")
