from abc import ABC, abstractmethod

import numpy as np

from adversarial_lab.core.tensor_ops import TensorOps

from typing import Literal, Any


class NoiseGenerator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply_noise(self,
                    sample: Any,
                    *args,
                    **kwargs
                    ) -> Any:
        pass

    def set_framework(self,
                      framework: Literal["tf", "torch", "numpy"]
                      ) -> None:
        if framework not in ["tf", "torch", "numpy"]:
            raise ValueError(
                "framework must be either 'tf', 'torch' or 'numpy'")
        self.framework = framework
        self.tensor_ops = TensorOps(framework)

        self._mask = self.tensor_ops.tensor(
            self._mask) if self._mask is not None else None
