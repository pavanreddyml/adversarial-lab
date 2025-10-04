from ...attacks_base import AttacksBase

from adversarial_lab.attacker.adversarial import WhiteBoxMisclassificationAttack
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator
from adversarial_lab.core.losses import CategoricalCrossEntropy, BinaryCrossEntropy
from adversarial_lab.core.optimizers import PGD
from adversarial_lab.core.constraints import POClip

from adversarial_lab.core.types import ModelType, TensorType
from typing import Callable, Optional, Union, Literal


class SmoothFoolNoiseGenerator(AdditiveNoiseGenerator):
    def __init__(self,
                 overshoot: float = 0.1,
                 sigma: float = 1.0,
                 kernel_size: int = 3,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.overshoot = overshoot
        self.sigma = sigma
        self.kernel_size = kernel_size

    def update(self,
               noise_meta,
               optimizer,
               grads,
               jacobian,
               logits,
               predictions,
               target_vector,
               true_class,
               *args,
               **kwargs
               ) -> None:
        noise = noise_meta[0]

        jacobian_tensor = jacobian
        logits_vector = self.tensor_ops.remove_batch_dim(logits, axis=0)
        predictions = self.tensor_ops.remove_batch_dim(predictions, axis=0)
        target_vector = self.tensor_ops.remove_batch_dim(target_vector, axis=0)

        grad_orig = jacobian_tensor[true_class]

        w = self.tensor_ops.zeros_like(noise)

        min_distance = float('inf')
        for i, jacobian in enumerate(jacobian_tensor):
            if jacobian is None or i == true_class:
                continue
            logit_val = logits_vector[i]
            w_k = jacobian - grad_orig
            f_k = (logit_val - logits_vector[true_class])
            distance = self.tensor_ops.abs(f_k) / (self.tensor_ops.norm(self.tensor_ops.reshape(w_k, [-1])) + 1e-8)
            distance = float(distance)

            if distance < min_distance:
                min_distance = distance
                w = w_k

        w_smoothed = self.tensor_ops.gaussian_blur(w, sigma=self.sigma, kernel_size=self.kernel_size)

        r_i = (min_distance + 1e-4) * w_smoothed / (self.tensor_ops.norm(self.tensor_ops.reshape(w_smoothed, [-1])) + 1e-8)

        noise.assign(noise + ((1 + self.overshoot) * r_i))


class SmoothFoolAttack(AttacksBase):
    def __init__(self,
                 model,
                 preprocessing_fn,
                 epsilon=0.1,
                 overshoot=0.1,
                 sigma=1.0,
                 kernel_size=3,
                 binary=False,
                 verbose=2,
                 *args,
                 **kwargs):
        self.attacker = WhiteBoxMisclassificationAttack(
            model=model,
            loss=CategoricalCrossEntropy() if not binary else BinaryCrossEntropy(),
            optimizer=PGD(learning_rate=0.1),
            noise_generator=SmoothFoolNoiseGenerator(requires_jacobian=True, overshoot=overshoot, sigma=sigma, kernel_size=kernel_size),
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
