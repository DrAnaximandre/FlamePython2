import numpy as np
from LFOs.LFOSet import LFOSet
from dataclasses import dataclass
from Variation import Variation
from colors import Color
import numpy as np

from Additives import spherical, linear, bubble, swirl, expinj, sinmoche, pdj, raton

@dataclass
class FunctionMapping():

    l: LFOSet = None
    colors: list = None
    weights: list = None
    additives: list = None
    Ax: np.ndarray = None
    Ay: np.ndarray = None
    probabilites: list = None

    def __post_init__(self):
        assert self.Ax.shape == self.Ay.shape
        assert self.Ax.shape[1] == 3
        assert self.Ax.shape[0] == len(self.weights)+1 # assuming the last function is the final
        assert len(self.weights)+1 == len(self.additives)
        assert len(self.colors) == len(self.weights)
        assert len(self.probabilites) == len(self.weights)
        self.construct()

    def call(self, coordinates):

        N_loc = coordinates.shape[0]  # how many points in the batch
        r = np.random.uniform(size=N_loc)  # each point is attributed a rand
        res = np.zeros(shape=(N_loc, 5))  # creation of the empty results x y r g b
        coordinates = np.concatenate((coordinates, 255 * np.ones((N_loc, 3))), axis=1)
    
        for i in range(len(self.relative_probabilites)-1):  # for each function
            # we select via a mask the points that are attributed a given function
            mask_1 = r > self.relative_probabilites[i]
            mask_2 = r < self.relative_probabilites[i + 1]
            selection = np.where((mask_1) & (mask_2))[0]

            # function call
            res[selection, :] = self.apply_function(i, coordinates[selection, :])
        
        # final call
        res= self.apply_function(i, coordinates)
        

        return res
 
    def construct_probas(self):
        self.relative_probabilites = [0] + [sum(self.probabilites[:i+1]) for i in range(len(self.probabilites))]
        

    def construct(self):
        self.construct_probas()
       
    def apply_function(self, i, coordinates):
        N_points = coordinates.shape[0]
        intercepts = np.ones((N_points, 1))
        points = np.concatenate((intercepts, coordinates[:,:2]), axis=1)
        x_loc = np.dot(points, self.Ax[i,:])
        y_loc = np.dot(points, self.Ay[i,:])
        
        result_loc = np.zeros((N_points, 5))

        for j in range(len(self.weights[i])):
            
            result_loc[:,:2] += self.weights[i][j] * self.additives[i][j](x_loc, y_loc)
        
        for j in range(3):
            if i != len(self.colors)-1:
                
                result_loc[:, j+2] = self.colors[i][j] + coordinates[:, j+2]


        return result_loc
        

if __name__ == "__main__":

    l = LFOSet()
    c = Color()
    colors = [c.R[3], c.G[3], c.B[2]]
   
    weights = [[0.5, 0.2], [0.5], [0.1,0.001,0.4]]
    additives = [[linear, swirl], [bubble], [linear, swirl, spherical], [linear]]
    Ax = np.array([[1,0,l.alpha0],[1,1,0],[1,0,1],[1,l.alpha1,1]])
    Ay = np.array([[-1,0,l.alpha2],[1,-1,0],[1,0,1],[1,l.alpha3,1]])
    probabilites = [0.2, 0.3, 0.5]

    fm = FunctionMapping(l, colors, weights, additives, Ax, Ay, probabilites)

    fm.call(np.random.rand(250000,2))


