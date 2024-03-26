import numpy as np
from LFOs.LFOSet import LFOSet
from Variation import Variation


@dataclass
class VariationFactory:

    l: LFOSet = None
    N: int = 25000
    colors: list = None
    A: np.ndarray = None
.   variation: Variation = None
    functions = FunctionMapping()


    def __post_init__(self):
        self.populate()

    def populate(self):
   
        variation = Variation(N)
        for f in self.functions:
            variation.addFunction(f.weights, f.params, f.additives, f.relative_proba, f.color)
        variation.addFinal([1], np.array([0,0.99,0,0,0,0.99]), [raton])
