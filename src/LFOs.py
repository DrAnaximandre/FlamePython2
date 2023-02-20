from dataclasses import dataclass
import numpy as np
from scipy import signal



def LFO(t:float):
    """
    Low frequency oscillator
    Args:
        t: time

    Returns:

    """
    return np.sin(t * 2 * np.pi)


@dataclass
class sLFO(object):

    speed: float = 2 * np.pi
    phase: float = 0
    min: float = 0
    max: float = 1
    def __call__(self, t):
        return np.sin(self.phase + t * self.speed) * (self.max - self.min) / 2 + (self.max + self.min) / 2

@dataclass
class tLFO(object):
    """triangular LFO"""

    speed: float = 2 * np.pi
    phase: float = 0
    min: float = 0
    max: float = 1
    width: float = 0.5
    def __call__(self, t):
        triangle = signal.sawtooth(self.phase + self.speed * t, self.width)* (self.max - self.min) / 2 + (self.max + self.min) / 2
        return triangle

if __name__ == "__main__":

    M = 1
    m = 0

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(m, M)
    ax.set_xlim(0, 1)
    t = np.linspace(0, 1, 250)
    ax.plot(t, sLFO(min =m, max=M)(sLFO(speed=5*np.pi, min=0.5, max = 1.9)(t)))
    plt.show()