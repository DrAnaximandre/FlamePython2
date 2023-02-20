import os
from dataclasses import dataclass
from Fractal import FractalParameters, Fractal
from VariationHolder import VariationHolder


@dataclass
class ImageFromParameters:

    i: int
    n_im: int
    name: str
    save: bool
    burn: int
    niter: int
    N: int
    zoom: float
    x: float
    y: float
    angle: float



    def __post_init__(self):
        self.folder_name = f"../images/{self.name}/"
        self.create_folder(self.folder_name)
        self.vh = VariationHolder(float(self.i) / self.n_im, self.N)

    def create_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def generate(self,
                 coef_forget=1.503,
                 coef_intensity=1.8,
                 optional_kernel_filtering=False,
                 verbose: bool = False):
        F1P = FractalParameters(self.burn, self.niter, self.zoom, self.x, self.y, self.angle, verbose)

        F1 = Fractal(F1P, [self.vh])
        F1.run()
        out, bitmap = F1.toImage(
            1024,
            coef_forget=coef_forget,
            coef_intensity=coef_intensity,
            optional_kernel_filtering=optional_kernel_filtering)
        if self.save:
            out.save(f"{self.folder_name}{self.name}-{self.i}.png")
