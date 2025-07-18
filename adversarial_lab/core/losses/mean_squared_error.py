from typing import List

from . import Loss
from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.types import TensorType, LossType


class MeanSquaredError(Loss):
    """
    Compute the mean squared error loss.
    """
    def __init__(self,
                 penalties: List[Penalty] = None,
                 from_logits: bool = False,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(penalties)

    def calculate(self,
                  target: TensorType,
                  predictions: TensorType,
                  logits: TensorType,
                  noise: TensorType,
                  ) -> LossType:
        loss = self.tensor_ops.losses.mean_squared_error(target=target,
                                                         predictions=predictions)
        loss = self._apply_penalties(loss, noise)
        self.set_value(loss)
        return loss
