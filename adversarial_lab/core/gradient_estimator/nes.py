import numpy as np
from copy import deepcopy

from . import GradientEstimator
from adversarial_lab.core.losses import Loss

from typing import Callable, List, Literal


class NESGradientEstimator(GradientEstimator):
    def __init__(self,
                 scale: List[float | int] = (0, 255),
                 sigma: float = 0.001,
                 max_perturbations: int = 10000,
                 batch_size: int = 32,
                 num_samples: int = 100,
                 antithetic: bool = True) -> None:
        super().__init__(scale=scale)

        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if max_perturbations <= 0:
            raise ValueError("max_perturbations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_samples % 2 != 0 and antithetic:
            raise ValueError("num_samples must be even when using antithetic sampling")

        self.scale = scale
        self.sigma = sigma * (scale[1] - scale[0])
        self.max_perturbations = max_perturbations
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.antithetic = antithetic

    def calculate(self,
                  sample: np.ndarray,
                  noise: List[np.ndarray],
                  target_vector: np.ndarray,
                  predict_fn: Callable,
                  construct_perturbation_fn: Callable,
                  loss: Loss,
                  mask: np.ndarray = None,
                  *args,
                  **kwargs) -> List[np.ndarray]:

        grads_wrt_noise = []

        for i, n in enumerate(noise):
            grad_i = np.zeros_like(n)
            shape = n.shape

            mask_i = mask if mask is not None else np.ones_like(n)
            flat_mask = mask_i.ravel()
            total_elements = int(np.sum(flat_mask))
            if total_elements == 0:
                grads_wrt_noise.append(grad_i)
                continue

            usable_elements = min(total_elements, self.max_perturbations)
            scale = 1.0 / self.num_samples

            noise_flat = n.ravel()
            grad_flat = np.zeros_like(noise_flat)
            mask_flat = mask_i.ravel()
            idxs = np.flatnonzero(mask_flat)

            rng = np.random.default_rng()

            all_perturbed_samples = []
            all_perturbed_noises = []
            directions_list = []

            for sample_idx in range(0, self.num_samples, 2 if self.antithetic else 1):
                z = np.zeros_like(noise_flat)
                z[idxs] = rng.normal(size=idxs.size)
                if self.antithetic:
                    z_neg = -z
                    zs = [z, z_neg]
                else:
                    zs = [z]

                for direction in zs:
                    perturbed_noise = deepcopy(noise)
                    perturbed_noise[i] = perturbed_noise[i].copy()
                    perturbed_noise[i].ravel()[idxs] += self.sigma * direction[idxs]
                    perturbed_sample = sample + construct_perturbation_fn(perturbed_noise)
                    perturbed_sample = np.clip(perturbed_sample, self.scale[0], self.scale[1])
                    all_perturbed_samples.append(perturbed_sample)
                    all_perturbed_noises.append(perturbed_noise)
                    directions_list.append(direction)

            predictions = []
            for b_start in range(0, len(all_perturbed_samples), self.batch_size):
                b_end = b_start + self.batch_size
                predictions.extend(predict_fn(all_perturbed_samples[b_start:b_end]))

            for j, direction in enumerate(directions_list):
                loss_j = loss.calculate(
                    target=target_vector,
                    predictions=predictions[j],
                    logits=None,
                    noise=construct_perturbation_fn([all_perturbed_noises[j]])
                )
                grad_flat[idxs] += loss_j * direction[idxs]

            grad_flat[idxs] *= scale / self.sigma
            grad_i = grad_flat.reshape(shape)
            grads_wrt_noise.append(grad_i)

        return grads_wrt_noise
