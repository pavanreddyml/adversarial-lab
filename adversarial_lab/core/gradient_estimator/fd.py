import numpy as np
from copy import deepcopy

from . import GradientEstimator
from adversarial_lab.core.losses import Loss

from typing import Callable, List, Literal, Tuple


class FDGradientEstimator(GradientEstimator):
    def __init__(self,
                 scale: List[float | int] = (0, 255),
                 epsilon: float = 1e-5,
                 max_perturbations: int = 10000,
                 batch_size: int = 32,
                 block_size: int = 1,
                 block_pattern: Literal["sequential", "square", "random"] = "sequential") -> None:
        super().__init__(scale=scale)

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if max_perturbations <= 0:
            raise ValueError("max_perturbations must be positive")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        if block_pattern not in {"sequential", "square", "random"}:
            raise ValueError(f"Invalid block_pattern '{block_pattern}', must be one of: 'sequential', 'square', 'random'")

        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even since each block generates 2 samples (+ε, -ε)")

        self.epsilon = epsilon
        self.max_perturbations = max_perturbations
        self.batch_size = batch_size
        self.block_size = block_size
        self.block_pattern = block_pattern

    def _get_indices(self, shape, mask):
        flat_mask = mask.ravel()
        indices = np.flatnonzero(flat_mask)
        if self.block_pattern == "sequential":
            return indices
        elif self.block_pattern == "random":
            rng = np.random.default_rng()
            return rng.permutation(indices)
        elif self.block_pattern == "square":
            side = int(np.floor(np.sqrt(self.block_size)))
            selected_indices = []
            visited = set()

            ndim = len(shape)
            if ndim < 2:
                raise ValueError("Shape must have at least 2 dimensions for square block pattern")

            H, W = shape[-2], shape[-1]

            for idx in indices:
                if idx in visited:
                    continue

                idx_unravel = np.unravel_index(idx, shape)
                if ndim == 2:
                    h, w = idx_unravel
                    prefix = ()
                else:
                    *prefix, h, w = idx_unravel

                if h + side > H or w + side > W:
                    continue

                block = []
                for i in range(side):
                    for j in range(side):
                        coords = prefix + (h + i, w + j)
                        flat_idx = np.ravel_multi_index(coords, shape)
                        block.append(flat_idx)

                if any(i in visited or flat_mask[i] == 0 for i in block):
                    continue

                visited.update(block)
                selected_indices.extend(block)

                if len(selected_indices) >= self.max_perturbations:
                    break

            return np.array(selected_indices[:self.max_perturbations])
        else:
            raise ValueError(f"Invalid block_pattern: {self.block_pattern}")

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
        total_perturbations = 0

        for i, n in enumerate(noise):
            grad_i = np.zeros_like(n)
            shape = n.shape
            indices = self._get_indices(shape, mask if mask is not None else np.ones_like(n))

            batch_pos_samples = []
            batch_neg_samples = []
            batch_metadata: List[Tuple[int, List[int]]] = []

            for block_start in range(0, len(indices), self.block_size):
                if total_perturbations >= self.max_perturbations:
                    break

                block_indices = indices[block_start:block_start + self.block_size]
                actual_block_size = len(block_indices)
                total_perturbations += actual_block_size

                pos_noise = deepcopy(noise)
                neg_noise = deepcopy(noise)

                for idx in block_indices:
                    pos_noise[i].flat[idx] += self.epsilon
                    neg_noise[i].flat[idx] -= self.epsilon

                perturbed_pos = sample + construct_perturbation_fn(pos_noise)
                perturbed_neg = sample + construct_perturbation_fn(neg_noise)

                perturbed_pos = np.clip(perturbed_pos, self.scale[0], self.scale[1])
                perturbed_neg = np.clip(perturbed_neg, self.scale[0], self.scale[1])

                batch_pos_samples.append(perturbed_pos)
                batch_neg_samples.append(perturbed_neg)
                batch_metadata.append((i, block_indices))

                # Process batch if full or last
                if len(batch_pos_samples) >= self.batch_size // 2 or block_start + self.block_size >= len(indices):
                    all_samples = batch_pos_samples + batch_neg_samples
                    predictions = predict_fn(all_samples)

                    for b in range(len(batch_metadata)):
                        loss_pos = loss.calculate(
                            target=target_vector, predictions=predictions[b], logits=None,
                            noise=construct_perturbation_fn([pos_noise]))
                        loss_neg = loss.calculate(
                            target=target_vector, predictions=predictions[b + len(batch_metadata)], logits=None,
                            noise=construct_perturbation_fn([neg_noise]))
                        grad_val = (loss_pos - loss_neg) / (2 * self.epsilon)

                        _, block_idxs = batch_metadata[b]
                        for idx in block_idxs:
                            grad_i.flat[idx] = grad_val

                    batch_pos_samples.clear()
                    batch_neg_samples.clear()
                    batch_metadata.clear()

            grads_wrt_noise.append(grad_i)

        return grads_wrt_noise
