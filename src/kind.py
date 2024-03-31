import numpy as np
from LFOs.LFOSet import LFOSet
from dataclasses import dataclass
from colors import Color
import numpy as np
from PIL import Image
from Additives import spherical, linear, bubble, swirl, expinj, sinmoche, pdj, raton
from numba import njit
from moviepy.editor import *

import glob
from natsort import natsorted

from joblib import Parallel, delayed

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
        
        # final call
        if self.final:
            result = self.apply_final(result)

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

        for j in range(len(self.weights[-1])):
            result[:,:2] = self.weights[-1][j] * self.additives[-1][j](x_loc, y_loc)
        
        return result


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
        initial_coordinates = np.concatenate((initial_coordinates, 255 * np.zeros((self.total_N, 3))), axis=1)

        for _ in range(self.burn_steps):
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
    
                    idx_start = (i*self.total_N+j*self.N)
                    idx_end = (i*self.total_N+(j+1)*self.N)
                    result[idx_start: idx_end] =  fm.call(result[idx_start:idx_end])
                    
            return result

    def to_image(self, result, size):

        imgtemp = Image.new('RGB', (size, size), "black")
        bitmap = np.array(imgtemp).astype(float)
        intensity = np.zeros((size, size))

        result_xy = (size * (result[:,:2] + 1) / 2)
        result_xy = np.concatenate((result_xy, result[:,2:]), axis=1)
        result_xy = result_xy.astype("i8")

        conditions = np.zeros((result_xy.shape[0], 4), dtype='bool')
        conditions[:, 0] = result_xy[:, 0] < size
        conditions[:, 1] = result_xy[:, 0] > 0
        conditions[:, 2] = result_xy[:, 1] < size
        conditions[:, 3] = result_xy[:, 1] > 0

        goods = np.where(np.all(conditions, 1))[0]


        bitmap, intensity = self.renderImage(
            result_xy, bitmap, intensity, goods)

        nmax = np.amax(intensity)
        
        intensity = np.power(np.log(intensity + 1) / np.log(nmax + 1), 1)

        bitmap = np.uint8(bitmap * np.reshape(np.repeat(intensity,3), (size,size,3)))

        out = Image.fromarray(bitmap)
        return(out, bitmap)


    @staticmethod
    @njit(fastmath=True)
    def renderImage(result, bitmap, intensity, goods, coef_forget=1):
        ''' this renders the image
            '''
        cf1 = coef_forget + 1
        result[:, 2:] = result[:, 2:] * coef_forget / cf1
        for i in goods:
            ad0 = result[i, 0]
            ad1 = result[i, 1]
            bitmap[ad0, ad1] /= cf1
            bitmap[ad0, ad1] += result[i, 2:]
            intensity[ad0, ad1] += 1
        return bitmap, intensity




def do_video_with_image_from_parameters(size=512):
    fps = 25
    n_im = 5*fps
    name = "test"

    images_to_generate = [ImageHolder(
        i,
        n_im,
        name=name,
        size=size,
    ) for i in range(n_im + 1)]

    Parallel(n_jobs=-2)(
        delayed(images_to_generate[i].run)() for i in range(n_im + 1)
    )

    base_dir = os.path.realpath(f"../images/{name}/")
    file_list = glob.glob(f'{base_dir}/{name}*.png')
    file_list_sorted = natsorted(file_list, reverse=False)

    clips = [ImageClip(m).set_duration(1 / fps)
             for m in file_list_sorted]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(f"{base_dir}/{name}.mp4", fps=fps, codec="libx264")


@dataclass
class ImageHolder():
    
    i: int = 0
    n_im: float = 20
    size: int = 1000
    name: str = "test"
    
    def __post_init__(self):
        self.ratio = self.i/self.n_im
        self.folder_name = f"../images/{self.name}/"
        self.create_folder(self.folder_name)


        l = LFOSet(ratio=self.ratio)
        c = Color()
        colors = [c.B[0], c.B[3], c.B[2]]
    
        weights = [[0.55], [0.5,-0.2], [0.5,0.1],[1, 0.2]]
        additives = [[linear], [linear,bubble], [linear, spherical], [linear, bubble]]
        Ax = np.array([[0,1,0],[1,1,0],[0,1,0],[0,1,0]])
        Ay = np.array([[0,0,1],[0,0,1],[1,0,1],[0,0,1]])
        probabilites = np.array([0.1, 0.1, 0.1])
        
        fm = FunctionMapping(l, colors, weights, additives, Ax, Ay, probabilites, final=False)

        self.vh = VH([fm],15,30, 5000)

    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def run(self):
        result = self.vh.run()
        out, _ = self.vh.to_image(result, 1000)
        out.save(f"{self.folder_name}{self.name}-{self.i}.png")

if __name__ == "__main__":

    #do_video_with_image_from_parameters()
    from time import time
    times = []
    for i in range(20):
        t0 = time()
        ih = ImageHolder(i)
        ih.run()
        t1 = time()
        times.append(t1-t0)

    print(f"Mean time: {np.mean(times)}")
       