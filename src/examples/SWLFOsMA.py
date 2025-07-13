from FunctionMapping import FunctionMapping
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical,pdj
from ImageHolder import ImageHolder
from Variation import Variation

class SWLFOsMA(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colorsA = [c.B[0], c.B[2], c.R[0], c.RNP[0], c.R[1]]
        colorsB = [c.B[2], c.B[1], c.R[1], c.RNP[1], c.B[0]]

        colors = l.alpha6 * np.array(colorsA) + ((1-l.alpha6) * np.array(colorsB))
    
        weights = [[0.7,0.65+l.gamma4/10,0.1],
                    [0.575,0.2+l.alpha4/10], 
                    [0.5,0.1-l.beta1/10],
                    [0.6],
                    [l.beta10*0.6],
                    [0.5+0.5*l.gamma0,0.3+0.5*l.alpha0]]
        additives = [[linear, bubble, spherical],
                    [linear, bubble],
                    [linear,bubble],
                    [bubble],
                    [bubble],
                    [linear, bubble]]
        Ax = np.array([[-1,-1,0.5*l.gamma1*l.alpha3],
                       [0,1,0.2],
                       [-1,0.5+0.5*l.beta2,0.1],
                       [1*l.gamma5,0.2,0.5*l.alpha5],
                       [0.1*l.beta3,0.3,l.alpha3],
                       [-0.3,2,-0.4]])
        Ay = np.array([[-l.alpha5/4,l.alpha5/2,-1],
                       [-1, 0,-1+l.gamma2/3],
                       [1,0,1+l.alpha4*l.beta1],
                       [1*l.alpha5,0.2,0.5*l.gamma6],
                        [0.3,l.gamma3,0.1*l.beta4],
                       [0.1,0.3,1]])
        probabilites = np.array([1+l.beta0, 1+l.alpha0, l.gamma0+1,0.9,1])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=False)
                      
        variation_params = ([fm], 20, 35, 300000)
        return Variation(*variation_params)


