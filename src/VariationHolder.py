from dataclasses import dataclass
from functools import partial

import numpy as np

from Additives import spherical, linear, bubble, swirl, expinj, sinmoche
from Variation import Variation
from LFOs import sLFO, tLFO


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


R = np.roll(np.array([cr1, cr2, cr3, cr4, cr5, cr6]),3)
B = np.roll(np.array([cb1, cb2, cb3, cb4, cb5, cb6]),0)
G = np.array([cg1, cg2, cg3])


@dataclass
class VariationHolder:

    """
    An object that holds one variation
    """

    i_rat: float = 0
    N: int = 25000
    M: int = 3
    kind: str = "demo"

    def __post_init__(self):
        self.variation = None
        self.generate()

    def generate(self):
        v = Variation(self.N)

        alpha = tLFO(min=0, max=0.25, speed=8*np.pi, phase=3*np.pi/2)(self.i_rat)
        gamma = tLFO(min=0, max=1, speed=2*np.pi, phase=0)(self.i_rat)
        beta = 2-tLFO(min=0, max=1, phase=0, width=0.75, speed=4 * np.pi)(self.i_rat) + tLFO(min=0, max=0.5, phase=np.pi/2, width=0.2175, speed=2 * np.pi)(self.i_rat)
        delta = 0.5+tLFO(min=-0.03, max=0.03, phase=3*np.pi/2, speed=16 * np.pi)(self.i_rat)
        theta = tLFO(min=0, max=1, phase=3*np.pi/2, width=0.155)(self.i_rat)- tLFO(min=0, max=0.025, phase=np.pi/2, width=0.171175, speed=16 * np.pi)(self.i_rat)
        tau = tLFO(min=0.5, max=1, phase=3*np.pi/2, width=0.256125, speed=8*np.pi)(self.i_rat)
        peta = tLFO(min=0.93, max=1.07, phase=3*np.pi/2, speed=2 * np.pi)(self.i_rat)
        zeta = 1-tLFO(min=0, max=1, phase=3*np.pi/2, speed=4 * np.pi, width=0.5)(self.i_rat)**2

        A = np.ones((self.M, 6))
        A[0, :] = [1, 1, alpha, 0, beta, 1]
        A[1, :] = [1-theta, alpha, 1, beta, 1, alpha]
        A[2, :] = [1-alpha, 0, 1*tau, beta*beta, alpha, 1*gamma]

        A = np.sin(5*A*peta*delta)+1-zeta


        for t in range(self.M):

            omega_t = tLFO(min=0.251, max=0.95, speed=16 * np.pi, width=.25, phase=t)(self.i_rat)
            lambda_t = tLFO(min=0.25, max=0.298,  width=0.165, speed=16 * np.pi, phase=t)(self.i_rat)
            eta_t = tLFO(min=0.1, max=0.1825, width=0.125, speed=2 * np.pi, phase=t)(self.i_rat)

            ct = alpha * (B[t%len(B)]) + (1 - alpha) * (R[t%len(R)])
            if t%3 ==0:
                ct = np.mean((G[t%len(G)],ct),0)

            epsilon_t = sLFO(min=0.0102, max=0.102, phase=t*3, speed=2 * np.pi)(self.i_rat)
            kappa_t = tLFO(min=0, max=0.0212, phase=t, speed=2 * np.pi, width=0.65)(self.i_rat)
            mu_t = tLFO(min=0, max=0.21075, width=0.7195, speed=2 * np.pi, phase=-t)(self.i_rat)

            if t%4!=0:
                v.addFunction([0.5, 0.001*mu_t*delta, 0.1*kappa_t],
                                  A[t, :],
                                  [linear, spherical, expinj],
                                  1 / self.M,
                                  ct)
            else:

                v.addFunction(
                    [1,0.002],
                    A[t, :],
                    [linear, bubble],
                    1 / self.M,
                    ct)

        #
        lda = [0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,0.33,0.0515,
                0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56,0.25,0.56]
        
        v.addRotation((32,2*np.pi/8,lda[:32]))


        hh = np.mean(A,0)
        hh = np.roll(hh, 1)
        hh = -3*np.tanh(hh)

        #v.addFinal([0.19,0.001,lambda_t*eta_t],hh,[spherical, bubble, linear])
        self.variation = v




