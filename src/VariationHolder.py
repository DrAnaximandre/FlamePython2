from dataclasses import dataclass
from functools import partial

import numpy as np
from PIL import Image
from Variation import Variation


from LFOs.LFOs import sLFO, tLFO



from Additives import spherical, linear, bubble, swirl, expinj, sinmoche, pdj, raton
#from LFOs import sLFO, tLFO


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


@dataclass
class VariationHolder:

    """
    An object that holds one variation
    """

    i_rat: float = 0
    N: int = 25000
    kind: str = "demo"

    def __post_init__(self):
        self.variation = None
        self.generate()

    def generate(self):

        R = np.roll(np.array([cr1, cr6, cr3, cr5, cr4, cr2]), 3)
        B = np.array([cb1, cb2, cb3, cb4, cb5, cb6])
        G = np.array([cg1, cg2, cg3, cg4, cg5, cg6])

        C = np.concatenate((R, B, G), axis=0)
        alpha = sLFO(max=0.5,speed=16*np.pi,min=0)(self.i_rat)
        alpha1 = sLFO(max=1,min=0, speed=8*np.pi,phase=1)(self.i_rat)
        alpha2 = sLFO(max=2,min=0, speed=4*np.pi,phase=2)(self.i_rat)
        alpha3 = sLFO(max=3,min=0, speed=2*np.pi,phase=3)(self.i_rat)

        beta = tLFO(phase=2,width=0.787,min=0.8,max=1.1)(self.i_rat) + tLFO(speed=4*np.pi,phase=0,width=0.187,max=0.12)(self.i_rat)
        beta0 = tLFO(phase=0,width=0.787,min=0.8,max=1.1)(self.i_rat) + tLFO(speed=4*np.pi,phase=2,width=0.187,max=0.12)(self.i_rat)
        gamma = sLFO(phase=1, speed=2*np.pi, max=1, min=0)(self.i_rat) - tLFO(speed=4*np.pi, max=0.3155, min=0.214)(self.i_rat)
        theta = sLFO(phase=4, speed=2*np.pi, max=0.55, min=0.215)(self.i_rat) * tLFO(speed=2*np.pi, width=0.75, max=0.3155, min=0.214)(self.i_rat)
        delta = sLFO(speed=4*np.pi, phase=0.3, max=0.93155, min=0.878214)(self.i_rat)

        kappa = tLFO(phase=0, width=0.5, max=1)(self.i_rat)


        v = Variation(self.N)
        if self.kind == "interpolation":
            np.random.seed(10)
            A = np.random.rand(6,6)*2-1
            for i in range(6):
                b = A[i,:] * beta + 2*A[(i+2)%A.shape[0],:] * (1-beta)
                v.addFunction([theta,0.55-theta,0.25*delta], b, [bubble, spherical,linear], 0.25, C[i*i%3])

        elif self.kind == "demo8":

            A = alpha * np.eye(6) + beta * np.ones((6,6)) + gamma * np.roll(np.eye(6),4,axis=0) + theta * np.roll(np.eye(6),2,axis=0) + delta * np.roll(np.eye(6),3,axis=0)
            A = 2*np.sin(A)

            c = [255,255,255]

            #A = (3-gamma) * np.sin(A*gamma*theta*delta) + 2*np.tanh(5*np.roll(A,1)*beta) - np.cos(5*np.roll(A,2)*delta)

            v.addFunction([1-beta,gamma*0.2], A[0,:], [spherical,linear], 1, c)
            v.addFunction([alpha,0.1], A[1,:], [linear, bubble], 1, [120,0,255])
            v.addFunction([gamma,0.75*gamma], A[2,:], [swirl, bubble], 1, [255,0,120])
            v.addFunction([0.5*beta,beta], A[3,:], [pdj, bubble], 1, c)

            for t in range(5):
                v.addFunction([0.5*beta*t,beta], np.roll(A[4,:],t), [swirl, linear], 1, c)

            nr = 3

            v.addRotation((nr*4,np.pi*2/nr,[1] * nr + [0.5]*nr + [0.3]*nr + [0.125]*nr))
            # v.addFinal([0.00025,0.72],
            #            [0.05,-1.02,0,-0.95,0.501,-1.01],
            #            [linear, bubble])


        elif self.kind == "today":

            a1 = beta*np.array([alpha1, -1, alpha2+0.5*alpha3, 1, 0, 1])
            a2 = beta*beta0*np.array([1, 1, alpha, 0, alpha2, 1])
            a3 = beta0*np.array([0, -1, alpha2, -1, -0.75, 0.25])
            a5 = np.array([-1, 1,alpha1,-0.75, -alpha1*alpha2*alpha3, 1])
            a8 = np.array([alpha2, -1, 0, 0, 1, 1])
            a9 = np.array([0, 1, alpha3, -1, beta0, 1])
            a4 = np.array([0, alpha2, 3, 0, -0.75, alpha1])
        

            # Define the color values as NumPy arrays
            c_1 = np.array([128, 0, 128])
            c_2 = np.array([255, 0, 255])
            c_3 = np.array([230, 230, 250])
            c_4 = np.array([128, 0, 0])
            c_5 = np.array([75, 0, 130])
            c_6 = np.array([0, 128, 128])
            c_7 = np.array([128, 0, 0])

            # Roll each array by 1
            c_1_rolled = np.roll(c_1, 1)
            c_2_rolled = np.roll(c_2, 1)
            c_3_rolled = np.roll(c_3, 1)
            c_4_rolled = np.roll(c_4, 1)
            c_5_rolled = np.roll(c_5, 1)
            c_6_rolled = np.roll(c_6, 1)
            c_7_rolled = np.roll(c_7, 1)


            
            v1 = Variation(self.N)
            v1.addFunction([.5-alpha+beta, 0.25 ], -a1, [spherical, bubble], .25, np.array(c_1)*alpha + np.array(c_1_rolled)*(1-alpha))
            v1.addFunction([.5-alpha1**2],-a2, [spherical], .25,np.array(c_2)*alpha + np.array(c_2_rolled)*(1-alpha))
            v1.addFunction([.5, alpha**3], -a4, [linear, bubble], .25, np.array(c_3)*alpha*alpha + np.array(c_3_rolled)*(1-alpha*alpha))
            v1.addFunction([alpha3**2,.75, alpha*0.00125], a5, [linear,spherical, bubble], .25, np.array(c_4)*alpha*alpha + np.array(c_4_rolled)*(1-alpha*alpha))
            v1.addFunction([.5*alpha3],a8, [bubble], .25, np.array(c_5)*alpha*alpha*alpha + np.array(c_5_rolled)*(1-alpha*alpha*alpha))
            v1.addFunction([.5, 0.725 * alpha2], a9, [bubble, spherical], .25, np.array(c_6)*alpha*alpha*alpha + np.array(c_6_rolled)*(1-alpha*alpha*alpha))
            v1.addFunction([.5,(0.5+alpha/2)/4], -a3, [linear, bubble], .25, np.array(c_7)*alpha + np.array(c_7_rolled)*(1-alpha))



#            v1.addFinal([0.150,0.95],
 #                                  [0.05+kappa/20,-1,alpha/15,0,0.00501+alpha/17,-1.01+delta/10],
  #                                 [ bubble, spherical])

            nr = 2

            self.variation = v1
            #self.variation = v

        elif self.kind == "test":

            a1 = np.array([0,1,0,0,0,1])
            a2 = np.array([1,1,0,0,0,1])
            a3 = np.array([0,1,0,1,0,1])

            v1 = Variation(self.N)

            v1.addFunction([0.55], a1, [linear], 0.1, B[0])
            v1.addFunction([0.5], a2, [linear], 0.1, B[3])
            v1.addFunction([0.5,0.1], a3, [linear, spherical], 0.1, B[2])

            self.variation = v1

        elif self.kind == "raton":

            a1 = beta*np.array([alpha1, -1, alpha2+0.5*alpha3, 1, 0, 1])
            a2 = beta*beta0*np.array([1, 1, alpha, 0, alpha2, 1])
            a3 = beta0*np.array([0, -1, alpha2, -1, -0.75, 0.25])
            a5 = 2*np.array([-1, 1,alpha1,-0.75, -alpha1*alpha2*alpha3, 1])
            a8 = np.array([alpha2, -1, 0, 0, 1, 1])
            a9 = np.array([0, 1, alpha3, -1, beta0, 1])
            a4 = np.array([0, alpha2, 3, 0, -0.75, alpha1])
        

            # Define the color values as NumPy arrays
            c_1 = np.array([128, 0, 128])
            c_2 = np.array([255, 0, 255])
            c_3 = np.array([230, 230, 250])
            c_4 = np.array([128, 0, 0])
            c_5 = np.array([75, 0, 130])
            c_6 = np.array([0, 128, 128])
            c_7 = np.array([128, 0, 0])

            # Roll each array by 1
            c_1_rolled = np.roll(c_1, 1)
            c_2_rolled = np.roll(c_2, 1)
            c_3_rolled = np.roll(c_3, 1)
            c_4_rolled = np.roll(c_4, 1)
            c_5_rolled = np.roll(c_5, 1)
            c_6_rolled = np.roll(c_6, 1)
            c_7_rolled = np.roll(c_7, 1)

            v1 = Variation(self.N)
            v1.addFunction([.5-alpha+beta, 0.925 ], alpha*a5-a9, [expinj, pdj], .25, np.array(c_1)*alpha + np.array(c_1_rolled)*(1-alpha))
            v1.addFunction([1,.5-alpha1**2],-a5, [pdj,spherical], .25,np.array(c_2)*alpha + np.array(c_2_rolled)*(1-alpha))
            v1.addFunction([0.4, alpha**3], [0,1,alpha,0,0,1], [bubble, swirl], .25, np.array(c_3)*alpha*alpha + np.array(c_3_rolled)*(1-alpha*alpha))
            v1.addFunction([alpha3**2,.575, alpha*0.00125], a9, [sinmoche,expinj, bubble], .25, np.array(c_4)*alpha*alpha + np.array(c_4_rolled)*(1-alpha*alpha))
            v1.addFunction([0.5*alpha3,1],a9, [bubble, expinj], .25, np.array(c_5)*alpha*alpha*alpha + np.array(c_5_rolled)*(1-alpha*alpha*alpha))
            v1.addFunction([.5, 0.725 * alpha2], a3, [sinmoche, spherical], .25, np.array(c_6)*alpha*alpha*alpha + np.array(c_6_rolled)*(1-alpha*alpha*alpha))
            v1.addFunction([0.5,(0.5+alpha3)], a8, [sinmoche, swirl], .25, np.array(c_7)*alpha + np.array(c_7_rolled)*(1-alpha))


            #v1.addFunction([1], np.array([alpha,1,beta,alpha2,alpha3,1]), [raton],0.25, [255,255,255])


            v1.addFinal([1], np.array([0,0.99,0,0,0,0.99]), [raton])



            self.variation = v1
