from latte.metrics.torch.disentanglement import *
from latte.metrics.torch.interpolatability import *

class LatteWrapper():
    def __init__(self) -> None:
        self.metrics = {
            "disentanglement": {
                "mig": MIG(),
                "dmig": DMIG(),
                "dlig": DLIG(),
                "xmig": XMIG(),
                "sap": SAP(), 
                "Modularity":  Modularity(),
            },
            "interpolatability": {
                "smoothness": Smoothness(),
                "monotonicity": Monotonicity(),
            }
        }

