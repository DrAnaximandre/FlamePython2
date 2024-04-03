from Function import Function
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical
from ImageHolder import ImageHolder
from Variation import Variation

class Sierpinskies(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colors = [c.R[0], c.R[3], c.RNP[2]]
    
        weights = [[0.5], [0.5], [0.5],[1]]
        additives = [[linear], [linear], [linear], [linear]]
        Ax = np.array([[-1,-1,0],[0,-1,0],[0,1,0],[0,1,0]])
        Ay = np.array([[0,0,-1],[-1,0,-1],[-1,0,1],[0,0,1]])
        probabilites = np.array([1, 1, 1])
        
        fm = Function(l, colors, weights, additives, Ax, Ay, probabilites, final=False)
                      
        variation_params = ([fm], 15, 30, 150000)
        return Variation(*variation_params)
