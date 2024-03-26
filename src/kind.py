import numpy as np
from LFOs.LFOSet import LFOSet
from Variation import Variation

@dataclass
class FuntionMapping():

    l: LFOSet = None
    colors: list = None
    wights: list = None
    A: np.ndarray = None

    def __post_init__(self):
        self.populate()

    def populate(self):
        self.functions = []
        for i in range(6):
            self.functions.append(Function([1], self.A[i], [self.l[i]]))



@dataclass
class VariationFactory:

    
    N: int = 25000
    func_map: FunctionMapping

    def __post_init__(self):
        self.populate()

    def populate(self):
   
        variation = Variation(N)
        for i, f in enumerate(self.functions):
            variation.addFunction(, f.params, f.additives, f.relative_proba, f.color)
        variation.addFinal([1], np.array([0,0.99,0,0,0,0.99]), [raton])
