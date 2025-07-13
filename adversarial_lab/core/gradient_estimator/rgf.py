import numpy as np
from copy import deepcopy

from . import GradientEstimator
from adversarial_lab.core.losses import Loss

from typing import Callable, List


class RGFGradientEstimator(GradientEstimator):
    def __init__(self,
                 scale: List[float | int] = (0, 255),
                 epsilon: float = 1e-3,
                 num_directions: int = 50,
                 max_perturbations: int = 10000,
                 batch_size: int = 32,
                 normalize: bool = True) -> None:
        super().__init__(scale=scale)

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if num_directions <= 0:
            raise ValueError("num_directions must be positive")
        if max_perturbations <= 0:
            raise ValueError("max_perturbations must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.scale = scale
        self.epsilon = epsilon * (scale[1] - scale[0])
        self.num_directions = num_directions
        self.max_perturbations = max_perturbations
        self.batch_size = batch_size
        self.normalize = normalize

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

            grad_flat = np.zeros_like(n.ravel())
            rng = np.random.default_rng()

            batch_samples = []
            direction_vectors = []

            for _ in range(self.num_directions):
                direction = np.zeros_like(n.ravel())
                direction[idxs] = rng.normal(size=usable_elements)

                if self.normalize:
                    direction /= np.linalg.norm(direction[idxs]) + 1e-10

                perturbed_noise = deepcopy(noise)
                perturbed_noise[i] = perturbed_noise[i].copy()
                perturbed_noise[i].ravel()[idxs] += self.epsilon * direction[idxs]

                perturbed_sample = sample + construct_perturbation_fn(perturbed_noise)
                perturbed_sample = np.clip(perturbed_sample, self.scale[0], self.scale[1])

                batch_samples.append(perturbed_sample)
                direction_vectors.append(direction)

                if len(batch_samples) >= self.batch_size:
                    predictions = predict_fn(batch_samples)
                    for j in range(len(batch_samples)):
                        loss_j = loss.calculate(
                            target=target_vector,
                            predictions=predictions[j],
                            logits=None,
                            noise=construct_perturbation_fn([deepcopy(noise)])
                        )
                        grad_flat[idxs] += loss_j * direction_vectors[j][idxs]

                    batch_samples.clear()
                    direction_vectors.clear()

            if batch_samples:
                predictions = predict_fn(batch_samples)
                for j in range(len(batch_samples)):
                    loss_j = loss.calculate(
                        target=target_vector,
                        predictions=predictions[j],
                        logits=None,
                        noise=construct_perturbation_fn([deepcopy(noise)])
                    )
                    grad_flat[idxs] += loss_j * direction_vectors[j][idxs]

            grad_flat[idxs] /= self.num_directions * self.epsilon
            grad_i = grad_flat.reshape(shape)
            grads_wrt_noise.append(grad_i)

        return grads_wrt_noise
