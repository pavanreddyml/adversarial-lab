from . import Loss
from adversarial_lab.core.penalties import Penalty
from adversarial_lab.core.types import TensorType, LossType

from typing import List


class DummyLoss(Loss):
    """
    Loss without any computation. Used for attacks that do not require a loss function like DeepFool, JSMA, etc or attacks that require the jacobian.
    """
    __dummy__ = True

    def __init__(self,
                 *args,
                 **kwargs
                 ) -> None:
        """
        Initialize the dummy loss function.

        Notes:
            value is set to 0 for tracking purposes.

        """
        self.value = 0

    def calculate(self, 
                  *args,
                  **kwargs
                  ) -> LossType:
        """
        Does not perform any computation.
        """
        return None
