

V1 = np.array([1, 1, 0, 1, 0, 1])
V2 = np.array([1, 0, -1, 0, 0, -1])
V3 = np.array([0, 1, 1, .5, 0, 1])
V4 = np.array([1, 0, 1, -1, 2, 1])
V5 = np.array([1, 0, 1, -1, 0, 0])

if __name__ == '__main__':

    def makeT1(N, i, burn=20, niter=50, zoom=1):

        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        v1.addFunction([-.5, .1], V1 + np.random.normal(0, .1, 6),
                       [linear, expinj], .45, [255, 0, 0])
        v1.addFunction([-.5, .5], V5 + np.random.normal(0, .1, 6),
                       [linear, expinj], .5, [0, 0, 255])
        v1.addFunction([.02, .5], V3 + np.random.normal(0, .1, 6),
                       [expinj, linear], .45, [2, 200, 25])
        v1.addFinal([.5, -.02], V5 + np.random.normal(0, .1, 6),
                    [linear, bubble])
        v1.addRotation((2 + np.random.randint(0, 30),
                        1 + np.random.normal(0, .2, 1)))
        F1.addVariation(v1, N)
        F1.build()
        F1.runAll()
        out = F1.toImage(64, coef=.5)
        out.save("t164/figure" + str(i) + ".png")

    def makeT2(N, i, burn=20, niter=50, zoom=1):

        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        v1.addFunction([-.5, .1], V2 + V1 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [255, 0, 0])
        v1.addFunction([-.5, .2], V3 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [0, 255, 0])
        v1.addFunction([.2, .5], V4 + np.random.normal(0, .1, 6),
                       [expinj, linear], .5, [0, 0, 255])
        v1.addFinal([.5, -.2], V1 + np.random.normal(0, .1, 6),
                    [linear, expinj])
        v1.addRotation((15 + np.random.randint(0, 15),
                        - 1 + np.random.normal(0, .2, 1)))
        F1.addVariation(v1, N)
        F1.build()
        F1.runAll()
        out = F1.toImage(64, coef=.5)
        out.save("t264/figure" + str(i) + ".png")

    def makeT3(N, i, burn=20, niter=50, zoom=1):

        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        v1.addFunction([-.5, .1], V2 - V1 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [255, 0, 0])
        v1.addFunction([-.5, .2], V3 - V4 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [0, 255, 0])
        v1.addFunction([.2, .5], V1 + V2 + V3 - V4 - V5 + np.random.normal(0, .1, 6),
                       [expinj, linear], .5, [0, 0, 255])
        v1.addFinal([.5, .2], V5 + np.random.normal(0, .1, 6),
                    [linear, expinj])
        v1.addRotation((0 + np.random.randint(0, 20),
                         0 + np.random.normal(0, .3, 1)))
        F1.addVariation(v1, N)
        F1.build()
        F1.runAll()
        out = F1.toImage(64, coef=.5)
        out.save("t364/figure" + str(i) + ".png")


    def makeT4(N, i, burn=20, niter=50, zoom=1):
        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        v1.addFunction([-.5, .1], V2 - V1 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [255, 0, 0])
        v1.addFunction([-.5, .2], V3 - V4 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [0, 255, 0])
        v1.addFunction([.2, .5], V1 + V2 + V3 - V4 - V5 + np.random.normal(0, .1, 6),
                       [expinj, linear], .5, [0, 0, 255])
        v1.addFinal([.5, .2], V5 + np.random.normal(0, .1, 6),
                    [bubble, pdj])
        v1.addRotation((0 + np.random.randint(0, 20),
                         0 + np.random.normal(0, .3, 1)))
        F1.addVariation(v1, N)
        F1.build()
        F1.runAll()
        out = F1.toImage(64, coef=.5)
        out.save("t464/figure" + str(i) + ".png")

    def makeT5(N, i, burn=20, niter=50, zoom=3):
        F1 = Fractale(burn, niter, zoom)
        v1 = Variation()
        v1.addFunction([-.5, .1], V1 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [255, 0, 0])
        v1.addFunction([-.5, .2], V1 + np.random.normal(0, .1, 6),
                       [linear, bubble], .5, [0, 255, 0])
        v1.addFunction([.2, .5], V1  + np.random.normal(0, .2, 6),
                       [expinj, linear], .5, [0, 0, 255])
        v1.addFinal([.5, .2], V1 + np.random.normal(0, .1, 6),
                    [spherical, pdj])
        v1.addRotation((0 + np.random.randint(0, 20),
                        0 + np.random.normal(0, .6, 1)))
        F1.addVariation(v1, N)
        F1.build()
        F1.runAll()
        out = F1.toImage(64, coef=.5)
        out.save("t564/figure" + str(i) + ".png")

    for i in range(6200):
        if i % 200 == 0:
            print(i)
        makeT5(100, i)
        # makeT2(100, i)
        # makeT3(100, i)
