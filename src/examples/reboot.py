from FunctionMapping import FunctionMapping
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical,pdj
from ImageHolder import ImageHolder
from Variation import Variation

class Reboot(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()

        colors = [c.CP[3], c.CP[2], c.CP[1], c.CP[0]]
     
        weights = [[0.65,0.5],[0.575,0.85],[0.5,0.3+l.beta5],[0.5],[0.495, l.beta3/10,l.alpha0/15]]
        additives = [[bubble,  linear],
                    [linear, spherical],
                    [linear, bubble],
                    [linear],
                    [linear, spherical,bubble],
                    ]
        Ax = np.array([[l.alpha0/2,1,0.05],
                       [0.51,1,0],
                       [1.1,-1,l.alpha2/4],
                       [-1,0.75-l.beta5,0.2],
                       [-0.725,1.2,-l.alpha5/5]])
        Ay = np.array([[-1,0.5,1],
                       [1,l.alpha3/12,1],
                       [0.5,-0.2,-0.1],
                       [1,-0.4,-1],
                       [-0.5,0,1.3]])
        probabilites = np.array([1,1,1,1])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=True)
                      
        variation_params = ([fm], 10, 40, 500000)
        return Variation(*variation_params)


