from dataclasses import dataclass
from src.Fractal import FractalParameters, Fractal
from src.ListOfVariations import ListOfVariations


@dataclass
class ImageFromParameters:

    i: int
    n_im: int = 0
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
        self.variations = [ListOfVariations(self.i,
                                            self.N,
                                            ) for i in range(self.n_im + 1)]

    def generate(self):
        F1P = FractalParameters(self.burn, self.niter, self.zoom, self.x, self.y, self.angle)

        F1 = Fractal(F1P, self.variations)
        F1.run()
        out, bitmap = F1.toImage(
            1024,
            coef_forget=1.003,
            coef_intensity=1.8,
            optional_kernel_filtering=False)
        if self.save:
            out.save(f"{self.folder_name}{self.name}-{self.i}.png")