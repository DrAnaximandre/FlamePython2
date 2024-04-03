from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from Variation import Variation


@dataclass
class ImageHolder(ABC):
    """
    This class represents an abstract base class for an image holder.
    It provides common attributes and methods for creating and saving images.

    Attributes:
        i (int): The current index of the image.
        n_im (float): The total number of images.
        size (int): The size of the image.
        name (str): The folder and name of the image.

    Methods:
        __post_init__(): Initializes the image holder.
        create_folder(folder_name): Creates a folder if it doesn't exist.
        create_variation(): Abstract method for creating a Variation
        run(): Runs the variation and saves the resulting image.
    """
    
    i: int = 0
    n_im: float = 20
    size: int = 1000
    name: str = "test"
    
    def __post_init__(self):
        self.ratio = self.i/self.n_im
        self.folder_name = f"../images/{self.name}/"
        self.create_folder(self.folder_name)
        self.variation = self.create_variation()

    @staticmethod
    def create_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    @abstractmethod
    def create_variation(self):
        pass

    def run(self):
        result = self.variation.run()
        out, _ = self.variation.to_image(result, self.size)
        out.save(f"{self.folder_name}{self.name}-{self.i}.png")

