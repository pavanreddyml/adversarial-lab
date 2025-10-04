from ...attacks_base import AttacksBase

from adversarial_lab.attacker.adversarial import WhiteBoxMisclassificationAttack
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy
from adversarial_lab.core.optimizers import Adam
from adversarial_lab.core.constraints import POClip
from adversarial_lab.core.losses import Loss
from adversarial_lab.core.penalties import LpNorm

from adversarial_lab.core.types import ModelType, TensorType

from typing import Callable, Optional, Union, Literal


class CWLoss(Loss):
    def __init__(self,
                 C: float = 0.0,
                 kappa: float = 0.0,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.C = C
        self.kappa = kappa

    def calculate(self,
                  target,
                  predictions,
                  logits,
                  noise,
                  *args,
                  **kwargs):
        target_logit = self.tensor_ops.reduce_sum(target * logits, axis=1)
        max_non_target_logit = self.tensor_ops.reduce_max(
            (1 - target) * logits - 1e4 * target,
            axis=1
        )

        f6 = self.tensor_ops.relu(
            max_non_target_logit - target_logit + self.kappa)
        loss = self.C * self.tensor_ops.reduce_mean(f6)
        loss = self._apply_penalties(loss, noise)

        self.set_value(loss)
        return loss


class CarliniWagnerAttack(AttacksBase):
    def __init__(self,
                 model,
                 preprocessing_fn,
                 C=0.0,
                 kappa=0.0,
                 learning_rate=0.01,
                 verbose=2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassificationAttack(
            model=model,
            loss=CWLoss(C=C, kappa=kappa, penalties=[LpNorm(p=2, lambda_val=1)]),
            optimizer=Adam(learning_rate=learning_rate),
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
