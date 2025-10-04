from ...attacks_base import AttacksBase

from adversarial_lab.attacker.adversarial import WhiteBoxMisclassificationAttack
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip


from adversarial_lab.core.types import ModelType, TensorType

from typing import Callable, Optional, Union, Literal


class ProjectedGradientDescentAttack(AttacksBase):
    def __init__(self,
                 model: ModelType,
                 preprocessing_fn: Callable,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 binary=False,
                 verbose: int = 2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassificationAttack(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=learning_rate),
            noise_generator=AdditiveNoiseGenerator(),
            preprocessing=preprocessing_fn,
            constraints=[POClip(min=-epsilon, max=epsilon)],
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
               epochs: int = 10,
               addn_analytics_fields: dict | None = None,
               *args,
               **kwargs):
        return self.attacker.attack(sample=sample,
                                    target_class=target_class,
                                    target_vector=target_vector,
                                    strategy=strategy,
                                    binary_threshold=binary_threshold,
                                    epochs=epochs,
                                    on_original=on_original,
                                    addn_analytics_fields=addn_analytics_fields,
                                    *args,
                                    **kwargs)

