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
cg4 = [20, 165, 20]
cg5 = [40, 200, 30]
cg6 = [60, 255, 5]


# some reds and purples
c_1 = np.array([128, 0, 128])
c_2 = np.array([255, 0, 255])
c_3 = np.array([230, 230, 250])
c_4 = np.array([128, 0, 0])
c_5 = np.array([75, 0, 130])
c_6 = np.array([0, 128, 128])


@dataclass
class Color:
    
    R = np.array([cr1, cr2, cr3, cr4, cr5, cr6])
    G = np.array([cg1, cg2, cg3, cg4, cg5, cg6])
    B = np.array([cb1, cb2, cb3, cb4, cb5, cb6])
    RNP = np.array([c_1, c_2, c_3, c_4, c_5, c_6])
    C = np.concatenate((R, G, B, RNP), axis=0)
