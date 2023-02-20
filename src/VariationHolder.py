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


R = [cr1, cr2, cr3, cr4, cr5, cr6]
B = [cb1, cb2, cb3, cb4, cb5, cb6]
G = [cg1, cg2, cg3]


@dataclass
class VariationHolder:

    """
    An object that holds one variation
    """

    i_rat: float = 0
    N: int = 25000
    M: int = 3

    def __post_init__(self):
        self.variation = None
        self.generate()

    def generate(self):
        v = Variation(self.N)

        alpha = tLFO(min=-0.2, max=1, phase=0)(self.i_rat)
        gamma = sLFO(min=0, max=0.58, speed=2 * np.pi, phase=np.pi/2.25)(self.i_rat)
        beta_t = tLFO(min=-0.5, max=0.5, phase=np.pi/2, speed=2 * 3.14)(self.i_rat)
        delta_t = tLFO(min=-0.02, max=0.02, phase=np.pi/2, speed=2 * np.pi)(self.i_rat)
        epsilon_t = sLFO(min=0.0, max=0.002, phase=0, speed=4 * 3.14)(self.i_rat)
        kappa_t = tLFO(min=0, max=1, phase=np.pi/2, speed=4 * 3.14)(self.i_rat)
        eta_t = tLFO(min=0, max=1, phase=0, width=0.25, speed=2 * 3.14)(self.i_rat)
        lambda_t = tLFO(min=0.5, max=0.55, phase=np.pi/2, width=0.85, speed=8 * 3.14)(self.i_rat)
        mu_t =  sLFO(min=0, max=0.25, speed=16 * np.pi, phase=np.pi/4)(self.i_rat)

        A = np.ones((self.M, 6)) * 0.102424

        A[0, :] = [-0.005, 0.951, eta_t, mu_t, gamma, 1]
        A[1, :] = [1, 0, 1, lambda_t, 1, beta_t]
        A[2, :] = [mu_t, alpha, kappa_t, 1, epsilon_t, 1]
        
        A *= 1.5

        for t in range(self.M):

            c1 = np.array(B[t%len(B)]) * eta_t + np.array(G[(t*2)%len(G)]) * (1-eta_t)
            c2 = np.array(B[t%len(B)])* kappa_t * (1-mu_t) + np.array(R[(t+1)%len(R)]) * (1-kappa_t)* (1-mu_t)
            c3 = np.array(R[(1+2*t)%len(R)]) * epsilon_t*100 + np.array(R[2]) * (1-epsilon_t*100)


            v.addFunction([0.5668+mu_t/2, delta_t, 0.0003*kappa_t],
                          A[t, :],
                          [linear, swirl, bubble],
                          1 / self.M,
                          (c1+c2)/2 if t%2 !=0  else c3)
        #v.addRotation((6,np.pi * 2 / 6,1))
        v.addFinal([kappa_t/100, 0.9932*lambda_t, -.927*gamma, 0.02591*alpha],
                    [0.005,1,-0.002,0.005,0,1],
                    [linear, spherical, bubble, swirl ])
        self.variation = v




