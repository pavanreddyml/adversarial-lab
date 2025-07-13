from abc import ABC, abstractmethod
from typing import Literal, List, Callable

from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.types import TensorType, TensorVariableType

class GradientEstimator(ABC):
    def __init__(self, scale) -> None:
        self.scale = scale

    @abstractmethod
    def calculate(self):
        pass
        
    def set_framework(self, 
                      framework: Literal["numpy"]
                      ) -> None:
        if framework not in ["numpy"]:
            raise ValueError("gradient estimator supports only 'numpy' framework")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

