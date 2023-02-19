from dataclasses import dataclass
from functools import partial

import numpy as np

from src.Additives import spherical, linear, bubble, swirl, expinj
from src.Variation import Variation
from LFOs import sLFO


@dataclass
class ListOfVariations:

    i_rat: float = 0
    N: int = 25000
    M: int = 3

    def __post_init__(self):
        self.list = []
        self.generate()

    def generate(self):
        v = Variation(self.N)

        alpha = sLFO(min=0.1, max=1)(self.i_rat)
        gamma = sLFO(min=0, max=1, speed=2 * np.pi, phase=1)(self.i_rat)

        A = np.ones((self.M, 6)) * 0.4
        for t in range(self.M):
            beta_t = sLFO(min=0.1, max=1, speed=4 * np.pi)(t / self.M)
            gamma_bt = sLFO(phase=np.sqrt(t + beta_t), speed=2 * np.pi, max=0.34999)(self.i_rat)

            for j in range(6):
                A[t, j] += np.where(j > 2 == 0, 1, -1.05) * np.cos(np.sqrt(j + alpha) + 2 * np.pi * t / self.M) * 0.810256
                A[t, j] += np.where(j % 3 == 0, 1, -1) * gamma_bt
                A[t, j] -= np.where(j % 5 == 0, -0.4, -0.4) * sLFO(min=0.5, max=0.6, phase=t)(self.i_rat)

        for t in range(self.M):

            delta_t = sLFO(min=0.2, max=0.3, phase=1+2*t, speed=2 * 3.14)(self.i_rat)
            print(delta_t)
            v.addFunction([0.5, delta_t],
                          A[t, :],
                          [linear, spherical],
                          1 / self.M,
                          [255,255*t/self.M,255*delta_t])
        self.list.append(v)




