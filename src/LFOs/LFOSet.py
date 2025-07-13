from LFOs.LFOs import sLFO, tLFO, cLFO
from dataclasses import dataclass
import numpy as np

@dataclass
class LFOSet():

    ratio: float

    def __post_init__(self):
        self.alpha0 = sLFO(max=1,min=0, speed=2*np.pi,phase=0)(self.ratio)
        self.alpha1 = sLFO(max=1,min=0, speed=8*np.pi,phase=1)(self.ratio)
        self.alpha2 = sLFO(max=1,min=0, speed=4*np.pi,phase=2)(self.ratio)
        self.alpha3 = sLFO(max=1,min=0, speed=2*np.pi,phase=3)(self.ratio)
        self.alpha4 = sLFO(max=1,min=0, speed=2*np.pi,phase=4)(self.ratio)
        self.alpha5 = sLFO(max=1,min=0, speed=2*np.pi,phase=5)(self.ratio)
        self.alpha6 = sLFO(max=1,min=0, speed=2*np.pi,phase=6)(self.ratio)
        self.alpha7 = sLFO(max=1,min=0, speed=2*np.pi,phase=7)(self.ratio)
        self.alpha8 = sLFO(max=1,min=0, speed=2*np.pi,phase=8)(self.ratio)
        self.alpha9 = sLFO(max=1,min=0, speed=2*np.pi,phase=9)(self.ratio)
        self.alpha10 = sLFO(max=1,min=0, speed=2*np.pi,phase=10)(self.ratio)

        self.beta0 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=0)(self.ratio)
        self.beta1 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=1)(self.ratio)
        self.beta2 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=2)(self.ratio)
        self.beta3 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=3)(self.ratio)
        self.beta4 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=4)(self.ratio)
        self.beta5 = tLFO(max=1,min=0, width=0.28, speed=2*np.pi,phase=5)(self.ratio)
        self.beta6 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=6)(self.ratio)
        self.beta7 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=7)(self.ratio)
        self.beta8 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=8)(self.ratio)
        self.beta9 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=9)(self.ratio)
        self.beta10 = tLFO(max=1,min=0, width=0.8, speed=2*np.pi,phase=10)(self.ratio)

        self.gamma0 = cLFO(max=1,min=0, speed=2*np.pi,phase=0)(self.ratio)
        self.gamma1 = cLFO(max=1,min=0, speed=2*np.pi,phase=1)(self.ratio)
        self.gamma2 = cLFO(max=1,min=0, speed=2*np.pi,phase=2)(self.ratio)
        self.gamma3 = cLFO(max=1,min=0, speed=2*np.pi,phase=3)(self.ratio)
        self.gamma4 = cLFO(max=1,min=0, speed=2*np.pi,phase=4)(self.ratio)
        self.gamma5 = cLFO(max=1,min=0, speed=2*np.pi,phase=5)(self.ratio)
        self.gamma6 = cLFO(max=1,min=0, speed=2*np.pi,phase=6)(self.ratio)

