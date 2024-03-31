import numpy as np
from LFOs.LFOSet import LFOSet
from dataclasses import dataclass
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
        
        # final call
        result = self.apply_final(result)
        

        return result
 
    def construct_probas(self):
        self.relative_probabilites = [0] + [sum(self.probabilites[:i+1]) for i in range(len(self.probabilites))]
        

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
            result_loc[:, j+2] = self.colors[i][j]/2 + coordinates[:, j+2]/2
           
        return result_loc
        
    def apply_final(self, result):

        points = self.build_points(result)
        x_loc = np.dot(points, self.Ax[-1,:])
        y_loc = np.dot(points, self.Ay[-1,:])

        for j in range(len(self.weights[-1])):
            result[:,:2] += self.weights[-1][j] * self.additives[-1][j](x_loc, y_loc)
        
        return result


if __name__ == "__main__":

    l = LFOSet(0.9)
    c = Color()
    colors = [c.R[3], c.G[3], c.B[2]]
   
    weights = [[0.5, 0.2], [0.5], [0.1,0.001,0.4],[0.3]]
    additives = [[linear, swirl], [bubble], [linear, swirl, spherical], [linear]]
    Ax = np.array([[1,0,l.alpha0],[1,1,0],[1,0,1],[1,l.alpha1,1]])
    Ay = np.array([[-1,0,l.alpha2],[1,-1,0],[1,0,1],[1,l.alpha3,1]])
    probabilites = [0.2, 0.3, 0.5]

    fm = FunctionMapping(l, colors, weights, additives, Ax, Ay, probabilites)


    l2 = LFOSet(0.5)
    weights = [[0.2, 0.5], [0.5], [0.21,0.2001,0.24],[0.23]]
    additives = [[linear, swirl], [bubble], [linear, swirl, spherical], [linear]]
    Ax = np.array([[1,0,l.alpha0],[1,1,0],[1,0,1],[1,l.alpha1,1]])
    Ay = np.array([[-1,0,l.alpha2],[1,-1,0],[1,0,1],[1,l.alpha3,1]])
    probabilites = [0.2, 0.3, 0.5]
    fm2 = FunctionMapping(l, colors, weights, additives, Ax, Ay, probabilites)

    @dataclass
    class VH():
        list_of_function_mappings: list
        burn_steps: int
        iterate_steps: int
        N: int

        def __post_init__(self):
            self.total_N = self.N*len(self.list_of_function_mappings)

        def run(self):
            initial_coordinates  = self.burn()
            result = self.iterate(initial_coordinates)
            return result

        def burn(self):
            
            initial_coordinates = np.random.uniform(-1,1, (self.total_N, 2))
            # it is the first time the coordinates are called, attach a color
            initial_coordinates = np.concatenate((initial_coordinates, 255 * np.ones((self.total_N, 3))), axis=1)
    
            for i in range(self.burn_steps):
                for j, fm in enumerate(self.list_of_function_mappings):
                    initial_coordinates[j*self.N:(j+1)*self.N,:] = fm.call(initial_coordinates[j*self.N:(j+1)*self.N,:])

            return initial_coordinates

        def iterate(self, initial_coordinates):
                
                result = np.zeros((self.total_N*self.iterate_steps, 5))

                for i in range(self.iterate_steps):
                    if i==0:
                        result[:self.total_N,:] = initial_coordinates
                    else:
                        result[i*self.total_N:(i+1)*self.total_N,:] = result[(i-1)*self.total_N:(i)*self.total_N,:]

                    for j, fm in enumerate(self.list_of_function_mappings):
                        result[i*self.total_N:(i+1)*self.total_N] = fm.call(result[i*self.total_N:(i+1)*self.total_N])
                       
                return result

        def to_image(self, result):

            pass

    vh = VH([fm, fm2],10,15, 2500)

    result = vh.run()
    print(result)