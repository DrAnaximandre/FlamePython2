from FunctionMapping import FunctionMapping
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical
from ImageHolder import ImageHolder
from Variation import Variation

class PixeF(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colorsA = [c.R[0], c.B[5], c.R[0], c.B[3]]
        colorsB = [c.B[2], c.G[5], c.B[1], c.G[2]]

        colors = l.alpha6 * np.array(colorsA) + ((1-l.alpha6) * np.array(colorsB))
       
    
        weights = [[0.65+l.alpha1/10,0.1], 
                   [0.75+l.alpha0/10,0.2+l.beta3/10],
                   [0.5],
                   [0.25+l.beta0/10],
                   [0.4,0.7]]
        additives = [[linear, bubble],
                     [linear, spherical],
                     [linear],
                     [linear],
                     [linear, bubble]]
        Ax = np.array([
            [-1,-1,0.5*l.gamma1/2*l.alpha3],
            [0,1,0],
            [1,0.5+0.5*l.beta2,0],
            [1,1,0],
            [1,0,0.05]])
        Ay = np.array([
            [0,0+l.alpha5,-1],
            [1, 0,-1-l.gamma2/3],
            [1,0,1+l.alpha3*l.beta1],
            [1,0,1],
            [0.4,-l.alpha4,1]])
        probabilites = np.array([1, 1, l.gamma0+1,-l.beta5])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=False)
                      
        variation_params = ([fm], 50, 20, 100000)
        return Variation(*variation_params)


