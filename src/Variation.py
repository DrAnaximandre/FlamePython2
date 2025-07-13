import numpy as np
from dataclasses import dataclass
from PIL import Image
from numba import njit

@dataclass
class Variation():
    list_of_function_mappings: list
    burn_steps: int
    iterate_steps: int
    N: int

    def __post_init__(self):
        self.total_N = self.N*len(self.list_of_function_mappings)

    def run(self):
        initial_coordinates  = self.burn()
        
        result = self.iterate(initial_coordinates)

        result = self.call_final(result)

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

    def call_final(self, result):
    
        for i in range(self.iterate_steps):
            for j, fm in enumerate(self.list_of_function_mappings):
                if fm.final:
                    idx_start = (i*self.total_N+j*self.N)
                    idx_end = (i*self.total_N+(j+1)*self.N)
                    result[idx_start: idx_end] =  fm.apply_final(result[idx_start:idx_end])
            
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
        
        intensity = np.power(np.log(intensity + 1) / np.log(nmax + 1), 0.725)

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

