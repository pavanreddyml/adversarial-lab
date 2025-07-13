import numpy as np
from copy import deepcopy

from . import GradientEstimator
from adversarial_lab.core.losses import Loss

from typing import Callable, List, Literal


class SPSAGradientEstimator(GradientEstimator):
    def __init__(self,
                 scale: List[float | int] = (0, 255),
                 epsilon: float = 0.01,
                 max_perturbations: int = 10000,
                 batch_size: int = 32,
                 num_samples: int = 128) -> None:
        super().__init__(scale=scale)

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if max_perturbations <= 0:
            raise ValueError("max_perturbations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.scale = scale
        self.epsilon = epsilon * (scale[1] - scale[0])
        self.max_perturbations = max_perturbations
        self.batch_size = batch_size
        self.num_samples = num_samples

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
            idxs = np.flatnonzero(flat_mask)

            if len(idxs) == 0:
                grads_wrt_noise.append(grad_i)
                continue

            usable_elements = min(len(idxs), self.max_perturbations)

            noise_flat = n.ravel()
            grad_flat = np.zeros_like(noise_flat)

            rng = np.random.default_rng()

            for batch_start in range(0, self.num_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_samples)
                batch_size_eff = batch_end - batch_start

                directions = []
                pos_samples = []
                neg_samples = []
                pos_noises = []
                neg_noises = []

                for _ in range(batch_size_eff):
                    delta = np.zeros_like(noise_flat)
                    delta[idxs] = rng.choice([-1, 1], size=usable_elements)

                    noise_pos = deepcopy(noise)
                    noise_neg = deepcopy(noise)
                    noise_pos[i] = noise_pos[i].copy()
                    noise_neg[i] = noise_neg[i].copy()
                    noise_pos[i].ravel()[idxs] += self.epsilon * delta[idxs]
                    noise_neg[i].ravel()[idxs] -= self.epsilon * delta[idxs]

                    sample_pos = sample + construct_perturbation_fn(noise_pos)
                    sample_neg = sample + construct_perturbation_fn(noise_neg)

                    sample_pos = np.clip(sample_pos, self.scale[0], self.scale[1])
                    sample_neg = np.clip(sample_neg, self.scale[0], self.scale[1])

                    directions.append(delta)
                    pos_samples.append(sample_pos)
                    neg_samples.append(sample_neg)
                    pos_noises.append(noise_pos)
                    neg_noises.append(noise_neg)

                predictions = predict_fn(pos_samples + neg_samples)

                for j in range(batch_size_eff):
                    loss_pos = loss.calculate(
                        target=target_vector,
                        predictions=predictions[j],
                        logits=None,
                        noise=construct_perturbation_fn([pos_noises[j]])
                    )
                    loss_neg = loss.calculate(
                        target=target_vector,
                        predictions=predictions[j + batch_size_eff],
                        logits=None,
                        noise=construct_perturbation_fn([neg_noises[j]])
                    )
                    g = (loss_pos - loss_neg) / (2 * self.epsilon)
                    grad_flat[idxs] += g * directions[j][idxs]

            grad_flat[idxs] /= self.num_samples
            grad_i = grad_flat.reshape(shape)
            grads_wrt_noise.append(grad_i)

        return grads_wrt_noise
