from . import PostOptimizationConstraint
from adversarial_lab.core.types import TensorVariableType, TensorType


class PONoisedSampleBounding(PostOptimizationConstraint):
    def __init__(self, min: float, max: float) -> None:
        if min > max:
            raise ValueError("min_val cannot be greater than max_val.")
        super().__init__()
        self.min_val = min
        self.max_val = max

    def apply(self, 
              noise: TensorVariableType,
              sample: TensorType,
              *args,
              **kwargs
              ) -> None:

        min_allowed_noise = self.tensor_ops.sub(self.min_val, sample)
        max_allowed_noise = self.tensor_ops.sub(self.max_val, sample)

        clipped_value = self.tensor_ops.clip(noise, min_allowed_noise, max_allowed_noise)
        self.tensor_ops.assign(noise, clipped_value)
