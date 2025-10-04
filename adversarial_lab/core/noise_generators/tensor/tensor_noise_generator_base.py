from abc import ABC, abstractmethod, ABCMeta

import numpy as np


from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.core.noise_generators import NoiseGenerator

from typing import Union, Optional, Callable, Any
from adversarial_lab.core.types import TensorType, TensorVariableType, OptimizerType


class TensorNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 mask: Optional[np.ndarray] = None,
                 requires_jacobian: bool = False) -> None:
        self._mask = mask
        self.requires_jacobian = requires_jacobian

    @abstractmethod
    def generate_noise_meta(self,
                            sample: Any,
                            *args,
                            **kwargs
                            ) -> Any:
        pass

    @abstractmethod
    def get_noise(self,
                  noise_meta: Any
                  ) -> np.ndarray:
        pass

    @abstractmethod
    def update(self,
               *args,
               **kwargs
               ) -> None:
        pass

    @abstractmethod
    def construct_noise(self,
                        noise_meta: Union[TensorVariableType],
                        *args,
                        **kwargs
                        ) -> Union[TensorVariableType, TensorType]:
        pass

    def apply_noise(self,
                    sample: TensorType | np.ndarray,
                    noise_meta: Union[TensorVariableType]
                    ) -> TensorType:
        return sample + self.construct_noise(noise_meta)

    def update(self,
               noise_meta: TensorVariableType,
               optimizer: OptimizerType | Optimizer,
               grads: TensorType,
               jacobian: TensorType = None,
               predictions: TensorType = None,
               target_vector: TensorType = None,
               true_class: int = None,
               predict_fn: Callable = None,
               *args,
               **kwargs
               ) -> None:
        optimizer.update(weights=noise_meta, gradients=grads)

    def get_mask(self) -> np.ndarray:
        return self.tensor_ops.numpy(self._mask) if self._mask is not None else None

    
