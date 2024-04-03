from Function import Function
from LFOs.LFOSet import LFOSet
from colors import Color
import numpy as np
from Additives import linear, bubble, spherical
from ImageHolder import ImageHolder
from Variation import Variation

class DemoImageHolder(ImageHolder):
    
    def create_variation(self):
        l = LFOSet(ratio=self.ratio)
        c = Color()
        colors = [c.B[0], c.B[3], c.RNP[2]]
    
        weights = [[0.55], [0.5,-0.2], [0.5,0.1],[1, 0.2]]
        additives = [[linear], [linear,bubble], [linear, spherical], [linear, bubble]]
        Ax = np.array([[-l.alpha2*0.05,1,l.alpha0/3],[1,1,0],[-l.alpha1*0.05,1,l.alpha2*0.05],[0,1,l.alpha1/10]])
        Ay = np.array([[l.alpha2/10,l.alpha3/10,1],[0,l.alpha3/20,1],[1,l.alpha0/10,1],[-l.alpha2/10,0,1]])
        probabilites = np.array([0.1, 0.1, 0.1])
        
        fm = Function(l, colors, weights, additives, Ax, Ay, probabilites, final=True)

        colors = [c.B[0], c.R[3], c.R[0]]
        weights = [[0.55], [0.5,-0.2], [0.5,0.1],[1, 0.2]]
        additives = [[linear], [linear,bubble], [linear, spherical], [linear, bubble]]
        Ax = np.array([[-1+l.alpha2*0.05,1,l.alpha0/3],[1,1,0],[0,1,0],[0,1,l.alpha1/10]])
        Ay = np.array([[l.alpha2/10,l.alpha3/10,1],[-1,l.alpha3/20,1],[1,l.alpha0/10,1],[-l.alpha2/10,0,1]])
        probabilites = np.array([0.1, 0.1, 0.1])
        
        fm2 = Function(l, colors, weights, additives, Ax, Ay, probabilites, final=True)

        variation_params = ([fm, fm2], 15, 30, 15000)
        return Variation(*variation_params)
