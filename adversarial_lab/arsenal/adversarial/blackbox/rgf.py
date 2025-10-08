from typing import Optional, Literal, Dict, Any

from ...attacks_base import AttacksBase

from adversarial_lab.attacker.adversarial import BlackBoxMisclassificationAttack
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import Adam, OptimizerRegistry
from adversarial_lab.core.constraints import POClip
from adversarial_lab.core.gradient_estimator import RGFGradientEstimator
from adversarial_lab.core.types import TensorType


class RGFAttack(AttacksBase):
    def __init__(
        self,
        pred_fn,
        image_scale: tuple = (0, 255),
        epsilon: float = 0.001,
        num_directions: int = 50,
        max_perturbations: int = 10000,
        batch_size: int = 32,
        normalize: bool = True,
        learning_rate: float = 1.0,
        verbose: int = 2,
        binary: bool = False,
        optimizer: Literal["adam", "pgd", "sgd"] = "adam",
        *args,
        **kwargs
    ):
        self.attacker = BlackBoxMisclassificationAttack(
            model=pred_fn,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=OptimizerRegistry.get(optimizer)(learning_rate=learning_rate),
            noise_generator=AdditiveNoiseGenerator(scale=image_scale, dist="zeros"),
            gradient_estimator=RGFGradientEstimator(scale=image_scale, epsilon=epsilon, max_perturbations=max_perturbations,
                                                    batch_size=batch_size, num_directions=num_directions,
                                                    normalize=normalize),
            constraints=[POClip(min=0, max=255)],
            verbose=verbose,
            *args,
            **kwargs
        )

    def attack(
        self,
        sample: TensorType,
        target_class: int = None,
        target_vector: TensorType = None,
        strategy: Literal['spread', 'uniform', 'random'] = "random",
        binary_threshold: float = 0.5,
        on_original: bool = False,
        epochs: int = 10,
        addn_analytics_fields: Dict[str, Any] | None = None,
        *args,
        **kwargs
    ):
        return self.attacker.attack(
            sample=sample,
            target_class=target_class,
            target_vector=target_vector,
            strategy=strategy,
            binary_threshold=binary_threshold,
            on_original=on_original,
            epochs=epochs,
            addn_analytics_fields=addn_analytics_fields,
            *args,
            **kwargs
        )
