from LFOs.LFOs import sLFO, tLFO
from dataclasses import dataclass
import numpy as np

@dataclass
class LFOSet():

    ratio: float = 0

    alpha0 = sLFO(max=1,min=0, speed=2*np.pi,phase=0)(ratio)
    alpha1 = sLFO(max=1,min=0, speed=8*np.pi,phase=1)(ratio)
    alpha2 = sLFO(max=1,min=0, speed=4*np.pi,phase=2)(ratio)
    alpha3 = sLFO(max=1,min=0, speed=2*np.pi,phase=3)(ratio)

