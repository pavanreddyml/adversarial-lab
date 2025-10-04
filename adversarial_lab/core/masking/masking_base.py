from abc import ABC, abstractmethod
from typing import Literal

from adversarial_lab.core.tensor_ops import TensorOps

class Masking(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def create(self, sample):
        pass

    def set_framework(self, 
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        if framework not in ["tf", "torch", "numpy"]:
            raise ValueError("framework must be either 'tf', 'torch', or 'numpy'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)