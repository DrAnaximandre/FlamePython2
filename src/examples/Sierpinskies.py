from FunctionMapping import FunctionMapping
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

        colors = [c.B[0], c.B[5], c.R[1]]
    
        weights = [[0.65], [0.75], [0.25],[1]]
        additives = [[linear], [linear], [linear], [linear]]
        Ax = np.array([[-1,-1,l.gamma1],[0,1,0],[1,1,0],[0,1,0]])
        Ay = np.array([[0,0,-1],[1, 0,-1],[1,0,1],[0,0,1]])
        probabilites = np.array([1, 1, 2])
        
        fm = FunctionMapping(l, 
                      colors, 
                      weights, 
                      additives, 
                      Ax, 
                      Ay, 
                      probabilites, 
                      final=False)
                      
        variation_params = ([fm], 15, 30, 150000)
        return Variation(*variation_params)
