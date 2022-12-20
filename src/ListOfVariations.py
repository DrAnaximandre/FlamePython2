from dataclasses import dataclass

from src.Variation import Variation


@dataclass
class ListOfVariations:

    i: int = 0
    N: int = 25000
    M: int = 5

    def __post_init__(self):
        self.list = []
        self.generate()

    def generate(self):
        for i in range(self.M):
            self.list.append(Variation(self.N))


