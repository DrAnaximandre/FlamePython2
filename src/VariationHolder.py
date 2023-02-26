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
    M: int = 5
    kind: str = "demo"

    def __post_init__(self):
        self.variation = None
        self.generate()

    def generate(self):
        v = Variation(self.N)

        alpha = sLFO(min=0, max=1, speed=2*np.pi, phase=np.pi/2)(self.i_rat)
        gamma = sLFO(min=0, max=1, speed=2 *np.pi, phase=0)(self.i_rat)
        beta = -tLFO(min=0, max=1, phase=0, width=0.75, speed=4 * np.pi)(self.i_rat) + tLFO(min=0, max=0.5, phase=np.pi/2, width=0.175, speed=2 * np.pi)(self.i_rat)
        delta = sLFO(min=0.25, max=0.5, phase=3*np.pi/2, speed=2 * np.pi)(self.i_rat)
        theta = tLFO(min=0, max=1, phase=np.pi / 3, width=0.155)(self.i_rat)
        tau = tLFO(min=-1, max=1, phase=0, width=0.6125)(self.i_rat)
        peta = sLFO(min=0.75, max=1, phase=np.pi / 2, speed=4 * np.pi)(self.i_rat)
        zeta = sLFO(min=0.5, max=0.75, phase=np.pi / 2, speed=4 * np.pi)(self.i_rat)

        A = np.ones((self.M, 6))
        A[0, :] = [-peta, 1, 0.25, zeta, beta, 1]
        A[1, :] = [1, gamma, zeta*beta, 0.5-zeta, theta, 1]
        A[2, :] = [zeta, 0.15+tau*tau, 1-delta, alpha, 0.25+delta, 1-alpha*peta]
        A[3, :] = [-1, theta*peta+beta, 1, -zeta, 1-tau, alpha]

        A[4, :] = [alpha, gamma, beta, delta, theta, tau]
        A = np.roll(A, 2, axis=1)
        A= np.tanh(A)


        for t in range(self.M):

            omega_t = tLFO(min=0.251, max=0.95, speed=2 * np.pi, width=.25, phase=t)(self.i_rat)
            lambda_t = tLFO(min=0.25, max=0.98,  width=0.165, speed=2 * np.pi, phase=t)(self.i_rat)
            eta_t = tLFO(min=0.1, max=0.1825, width=0.125, speed=2 * np.pi, phase=t)(self.i_rat)

            ct = alpha * (B[t%len(B)]) + (1 - alpha) * (R[t%len(R)])
            if t==0:
                ct = np.mean((G[0],ct),0)

            epsilon_t = sLFO(min=0.0102, max=0.102, phase=t*3, speed=2 * np.pi)(self.i_rat)
            kappa_t = tLFO(min=0, max=0.212, phase=t, speed=2 * np.pi, width=0.65)(self.i_rat)
            mu_t = tLFO(min=0, max=0.21075, width=0.7195, speed=2 * np.pi, phase=-t)(self.i_rat)

            if t!=0:
                v.addFunction([np.max((np.min((0.5,0.75-delta+eta_t)),0.25)), 0.1*mu_t*delta, kappa_t],
                                  A[t, :],
                                  [linear, bubble, spherical],
                                  1 / self.M,
                                  ct)
            else:

                v.addFunction(
                    [np.max((np.min((0.5+epsilon_t, 0.75 - lambda_t + eta_t)), 0.25)),0.2],
                    np.tanh(A[t, :]),
                    [linear, bubble],
                    1 / self.M,
                    ct)


        lda = [1,0.2,1,0.2,1,0.2,1,0.2,1,0.2,1,0.2,0.5,0.75,0.5,0.75,0.5,0.75,0.5,0.75,0.5,0.75,0.5,0.75]

        v.addRotation((24,2*np.pi/12,lda))


        hh = np.mean(A,0)
        hh = np.roll(hh, 1)
        hh = 3*np.tanh(hh)

        #v.addFinal([0.9,0.1+delta/5,0.1-delta/5],hh,[bubble, linear, spherical])
        self.variation = v




