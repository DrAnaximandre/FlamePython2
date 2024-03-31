from LFOs.LFOs import sLFO, tLFO
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

