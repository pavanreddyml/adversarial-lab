from ...attacks_base import AttacksBase

from adversarial_lab.attacker.adversarial import WhiteBoxMisclassificationAttack
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD

from adversarial_lab.core.types import ModelType, TensorType

from typing import Callable, Optional, Union, Literal


class FastSignGradientMethodAttack(AttacksBase):
    def __init__(self,
                 model: ModelType,
                 preprocessing_fn: Callable,
                 epsilon: float = 0.1,
                 binary=False,
                 verbose: int = 2,
                 *args,
                 **kwargs
                 ):
        self.attacker = WhiteBoxMisclassificationAttack(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=epsilon),
            noise_generator=AdditiveNoiseGenerator(),
            preprocessing=preprocessing_fn,
            verbose=verbose,
            *args,
            **kwargs
        )

    def attack(self,
               sample: TensorType,
               target_class: int = None,
               target_vector: TensorType = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               binary_threshold: float = 0.5,
               on_original: bool = False,
               addn_analytics_fields: dict | None = None,
               *args,
               **kwargs):
        kwargs.pop("epochs", None)
        return self.attacker.attack(sample=sample,
                                    target_class=target_class,
                                    target_vector=target_vector,
                                    strategy=strategy,
                                    binary_threshold=binary_threshold,
                                    epochs=1,
                                    on_original=on_original,
                                    addn_analytics_fields=addn_analytics_fields,
                                    *args,
                                    **kwargs)
