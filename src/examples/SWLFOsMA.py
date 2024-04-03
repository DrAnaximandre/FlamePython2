from FunctionMapping import FunctionMapping
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical
from ImageHolder import ImageHolder
from Variation import Variation

class SWLFOsMA(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colorsA = [c.B[0], c.B[5], c.R[0]]
        colorsB = [c.B[2], c.G[5], c.R[1]]

        colors = l.alpha6 * np.array(colorsA) + ((1-l.alpha6) * np.array(colorsB))
       
    
        weights = [[0.65+l.alpha1/10,0.1], [0.75+l.alpha0/10,0.2+l.beta3/10], [0.25+l.beta0/10],[1]]
        additives = [[linear, bubble], [linear, spherical], [linear], [linear]]
        Ax = np.array([[-1,-1,l.gamma1/2*l.alpha3],[0,1,0],[1,1,0],[0,1,0]])
        Ay = np.array([[0,0+l.alpha5,-1],[1, 0,-1-l.gamma2/3],[1,0,1+l.alpha3*l.beta1],[0,0,1]])
        probabilites = np.array([1, 1, l.gamma0+1])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=False)
                      
        variation_params = ([fm], 15, 30, 100000)
        return Variation(*variation_params)


