import numpy as np
from LFOs.LFOSet import LFOSet
from dataclasses import dataclass

@dataclass
class FunctionMapping():

    l: LFOSet = None
    colors: list = None
    weights: list = None
    additives: list = None
    Ax: np.ndarray = None
    Ay: np.ndarray = None
    probabilites: list = None
    final: bool = False

    def __post_init__(self):
        assert self.Ax.shape == self.Ay.shape
        assert self.Ax.shape[1] == 3
        assert self.Ax.shape[0] == len(self.weights) # assuming the last function is the final
        assert len(self.weights) == len(self.additives)
        assert len(self.colors) == len(self.weights)-1 # the final has no color
        assert len(self.probabilites) == len(self.weights)-1 # the final has no probability
        self.construct()

    def call(self, coordinates):

        N_loc = coordinates.shape[0]  # how many points in the batch
        r = np.random.uniform(size=N_loc)  # each point is attributed a rand
        result = np.zeros(shape=(N_loc, 5))  # creation of the empty results x y r g b
        
        for i in range(len(self.relative_probabilites)-1):  # for each function
            # we select via a mask the points that are attributed a given function
            mask_1 = r > self.relative_probabilites[i]
            mask_2 = r < self.relative_probabilites[i + 1]
            selection = np.where((mask_1) & (mask_2))[0]

            # function call
            result[selection, :] = self.apply_function(i, coordinates[selection, :])
        

        return result
 
    def construct_probas(self):
        self.relative_probabilites = [0] + [sum(self.probabilites[:i+1]) for i in range(len(self.probabilites))]
        self.relative_probabilites /= self.relative_probabilites[-1]
        

    def construct(self):
        self.construct_probas()
    
    @staticmethod
    def build_points(coordinates):
        N_points = coordinates.shape[0]
        intercepts = np.ones((N_points, 1))
        points = np.concatenate((intercepts, coordinates[:,:2]), axis=1)
        return points

    def apply_function(self, i, coordinates):
       
        points = self.build_points(coordinates)
        x_loc = np.dot(points, self.Ax[i,:])
        y_loc = np.dot(points, self.Ay[i,:])
        result_loc = np.zeros((points.shape[0], 5))

        for j in range(len(self.weights[i])):
            result_loc[:,:2] += self.weights[i][j] * self.additives[i][j](x_loc, y_loc)
        
        for j in range(3):
            result_loc[:, j+2] = self.colors[i][j] + coordinates[:, j+2]
            result_loc[:, j+2] /= 2
           
        return result_loc
        
    def apply_final(self, result):

        points = self.build_points(result)
        x_loc = np.dot(points, self.Ax[-1,:])
        y_loc = np.dot(points, self.Ay[-1,:])
        result_loc = np.zeros((points.shape[0], 5))
        result_loc[:,2:] = result[:,2:]

        for j in range(len(self.weights[-1])):
            result_loc[:,:2] += self.weights[-1][j] * self.additives[-1][j](x_loc, y_loc)
        
        return result_loc

