from class_fractale import *
import numpy as np
from pyDOE import *

def addV(offset):

    v1 = Variation(offset)
    
    if np.random.rand()< .5:
        nf = np.random.randint(4,7)
        naddm = 3
    else:
        nf = np.random.randint(25,60)
        naddm = 2

    Ws = lhs(naddm, samples=nf, criterion='center') * 2 - 1 + np.random.normal(0, sd, size=(nf,naddm))
    A = lhs(6, samples=nf, criterion='center') * 2 - 1 + np.random.normal(0, sd, size=(nf,6))
    for lig in range(nf):
        for z in range(np.random.random_integers(0,4)):
            A[lig,np.random.randint(0,6)] =0
    print(str(i) + " ---  hue : " + str(gethue))

    for f in range(nf):
        nadds = naddm
        ws = Ws[f,:]
        a = A[f, :] 
        adds = [linear]
        for t in range(nadds):
            if np.random.rand()<.5:
                adds.append(ladd0[np.random.randint(len(ladd0))])
            else :
                adds.append(ladd1[np.random.randint(len(ladd1))])

        gotohue = hue[f%5] + np.random.normal(0,sdcol,size=3).astype(int)
        gotohue[gotohue > 255] = 255
        gotohue[gotohue < 0] = 0
        v1.addFunction(ws, a, adds, np.random.rand()*3, gotohue )


    rr = np.random.rand()
    if rr < .2:
        v1.addRotation(
            (np.random.randint(15),
             np.random.uniform(0, 2),
             np.random.normal(1, .3)))
    elif rr < .4:
        v1.addRotation(
            (np.random.randint(20),
             np.random.uniform(-1, 0),
             np.random.normal(1, .3)))

    return(v1)

if __name__ == '__main__':

    # yo do not touch this
    burn = 20
    niter = 30
    N = 100000

    # ok now you can play with all the parameters
    zoom = 1

    sethue0 = [[233,157,37],
[216,209,53],
[70,105,230],
[28,69,133],
[67,117,206]] # jaune et bleus
    sethue1 =[[110,209,79],
[20,200,12],
[226,47,49],
[250,17,21],
[255,32,52]]# rouge et verts
    sethue2 = [[96,122,136],
[59,116,214],
[5,15,39],
[255,255,255],
[0,71,112]] # bleu sombre et blanc
    sethue3 = [[0,220,162],
[215,226,246],
[10, 10, 10],
[164,250,255],
[1,160,183]] #bleu clair et presque noir
    sethue4 = [[235,216,255],
[53,149,155],
[229,124,127],
[127,140,218],
[255,219,193]] #presque blanc
    sethue5=[[67,45,97],
[91,42,162],
[0,110,208],
[250,250,250],
[250,250,250]]
#violet et blanc
    sethue6=[[200,200,200],
[50,50,50],
[100,100,100],
[150,150,150],
[250,250,250]]
# gris
    sethue7=[[215,216,66],
[179,102,158],
[219,236,74],
[181,102,158],
[212,39,226]] #jaune laid et violet merdique

    sethue8=[[255,0,0],
[50,0,250],
[100,0,200],
[150,0,0],
[20,10,200]]
# rouges et bleus presque violet en fait
    hues = [eval("sethue" + str(i)) for i in range(8)]
    #hues = [sethue7]

    ladd0 = [linear]
    ladd1 = [swirl, spherical, expinj,bubble]

    sd = .5
    sdcol = 15
    #sdw = 1

    for i in range(15):
        gethue = np.random.randint(len(hues))
        hue = hues[gethue]
        np.random.shuffle(hue)

        F1 = Fractale(burn, niter, zoom)

        F1.addVariation(addV([0,0]), N)
        F1.addVariation(addV([.25,.25]), N)
        F1.addVariation(addV([-.75,-.75]), N)

        # nope! cant touch this
        # stop playing with the parameters now
        F1.build()
        F1.runAll()
        rescore = F1.toScore()
        f = open("doe/doe.txt", 'a')
        f.write(str(i) + ";" + rescore + "\n")

        print("    Generating the image")
        out = F1.toImage(1000, coef_forget=.05,
                         coef_intensity=.22,
                         optional_kernel_filtering=False)
        out.save("doe/doe" + str(i) + ".png")
