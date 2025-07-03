import numpy as np

from .tensor_noise_generator_base import TensorNoiseGenerator

from typing import Literal, List, Union
from adversarial_lab.core.types import TensorType, TensorVariableType


class AdditiveNoiseGenerator(TensorNoiseGenerator):
    def __init__(self,
                 scale: List[int] = (-1, 1),
                 dist: Literal["zeros", "ones", "normal", "uniform"] = "zeros",
                 mask: TensorType | np.ndarray = None,
                 requires_jacobian: bool = False,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(mask=mask, 
                         requires_jacobian=requires_jacobian)

        self.dist = dist
        self.scale = scale

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, TensorType],
                            ) -> TensorVariableType:
        if self.tensor_ops.has_batch_dim(sample):
            unbatched_sample = self.tensor_ops.remove_batch_dim(sample)
        else:
            unbatched_sample = sample
        unbatched_sample_shape = unbatched_sample.shape

        if self.dist == "zeros":
            noise_meta = self.tensor_ops.zeros_like(unbatched_sample, unbatched_sample.dtype)
        elif self.dist == "ones":
            noise_meta = self.tensor_ops.ones_like(unbatched_sample, unbatched_sample.dtype)
        elif self.dist == "normal":
            noise_meta = self.tensor_ops.random_normal(unbatched_sample_shape)
            noise_meta = self.scale[0] + (self.scale[1] - self.scale[0]) * (noise_meta - self.tensor_ops.reduce_min(
                noise_meta)) / (self.tensor_ops.reduce_max(noise_meta) - self.tensor_ops.reduce_min(noise_meta))
        elif self.dist == "uniform":
            noise_meta = self.tensor_ops.random_uniform(
                unbatched_sample_shape, minval=self.scale[0], maxval=self.scale[1])
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")
        
        self._mask = self.tensor_ops.ones_like(noise_meta) if self._mask is None else self.tensor_ops.tensor(self._mask)

        noise_meta = self.tensor_ops.variable(noise_meta)

        self.tensor_ops.assign(noise_meta, noise_meta * self._mask)
        return [noise_meta]

    def get_noise(self,
                  noise_meta: List[TensorVariableType]
                  ) -> np.ndarray:
        return self.tensor_ops.numpy(noise_meta[0])

    def construct_noise(self,
                    noise_meta: List[TensorVariableType]
                    ) -> TensorType:
        return self._mask * noise_meta[0]
