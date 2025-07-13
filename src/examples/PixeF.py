from FunctionMapping import FunctionMapping
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical, expinj
from ImageHolder import ImageHolder
from Variation import Variation

class PixeF(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colorsA = [c.R[0], c.B[5], c.R[0], c.B[3],c.R[5] ]
        colorsB = [c.B[2], c.R[5], c.B[1], c.G[2],c.B[4]]

        colors = [c.R[0], c.R[3], c.R[2]]
    
        weights = [[0.5], [0.5], [0.5],[1]]
        additives = [[linear], [linear], [linear], [bubble]]
        Ax = np.array([[l.alpha0,1,0],[1,1,0],[0,1,0],[l.beta2,1,l.beta7]])
        Ay = np.array([[0,0,1],[1,l.alpha3,1],[1,0,1],[-l.beta3,l.beta4*1.1,1]])
        probabilites = np.array([1, 1, 1])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=True)
                      
        variation_params = ([fm], 10, 30, 20000)
        return Variation(*variation_params)


