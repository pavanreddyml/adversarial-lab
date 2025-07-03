from . import Preprocessing
from adversarial_lab.core.types import TensorType

class NoPreprocessing(Preprocessing):
    def __init__(self) -> None:
        pass

    def preprocess(self, sample: TensorType) -> TensorType:
        return sample
    