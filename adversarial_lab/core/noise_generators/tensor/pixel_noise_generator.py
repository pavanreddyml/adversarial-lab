from typing import Literal, List, Union, TypeVar, Generic, Dict, Any, Tuple, Union
import numpy as np

from . import NoiseGenerator
from adversarial_lab.core.optimizers import Optimizer

import tensorflow as tf
import torch


class PixelNoiseGenerator(NoiseGenerator):
    def __init__(self,
                 scale: Tuple[int] = (-1, 1),
                 n_pixels: int = 1,
                 pixel_selector: Literal["positive", "negative", "absolute"] = "absolute",
                 direction: Literal["positive", "negative", "bi"] = "positive",
                 strategy: Literal["single", "iterative", "incremental"] = "single",
                 reduce_dims: List[int] = None,
                 epsilon: float = np.inf,
                 *args, 
                 **kwargs
                 ) -> None:
        self.n_pixels = n_pixels
        self.scale = scale
        self.pixel_selector = pixel_selector
        self.direction = direction
        self.strategy = strategy
        self.reduce_dims = reduce_dims
        self.epsilon = [-epsilon, epsilon]

        self.increment_per_iteration = kwargs.get("increment_per_iteration", 1)
        self._reset_optimization_meta()

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, torch.Tensor, tf.Tensor],
                            bounds: List[List[int]] = None
                            ) -> tf.Variable:
        super().generate_noise_meta(sample, bounds)

        shape = sample.shape
        if shape[0] is None or shape[0] == 1:
            shape = shape[1:]

        noise = tf.zeros_like(sample)
        noise = self.tensor_ops.variable(noise)

        self._reset_optimization_meta()

        noise.assign(noise * self._mask)
        self.apply_constraints(noise)

        return [noise]

    def get_noise(self,
                  noise_meta: List[tf.Tensor | tf.Variable]
                  ) -> np.ndarray:
        return tf.clip_by_value(noise_meta[0], self.scale[0], self.scale[1]).numpy()

    def apply_noise(self,
                    sample: tf.Tensor,
                    noise: tf.Variable
                    ) -> tf.Tensor:
        return tf.clip_by_value(sample + (noise[0] * self._mask), self.scale[0], self.scale[1])
    
    def apply_gradients(self,
                        noise: tf.Variable,
                        gradients: tf.Tensor,
                        optimizer: Optimizer = None) -> None:
        
        

        if self.strategy == "single":
            self._optimize_single(noise[0], gradients)
        elif self.strategy == "iterative":
            self._optimize_iterative(noise, gradients, optimizer)
        elif self.strategy == "incremental":
            self._optimize_incremental(noise[0], gradients)
        else:
            raise ValueError(f"Invalid Strategy: {self.strategy}")

    def _optimize_single(self,
                         noise: List[tf.Variable],
                         gradients: tf.Tensor,
                         ) -> None:
        masked_gradients = self._get_masked_gradients(gradients, noise[0])
        attacked_pixels = self._get_attacked_pixels(masked_gradients, noise[0])
        noise[0].assign(attacked_pixels)
        self.end_optimization = True

    def _optimize_iterative(self,
                            noise: tf.Variable,
                            gradients: tf.Tensor,
                            optimizer: Optimizer,
                            ) -> None:

        if self.optimization_meta["iterative_mask"] is None:
            masked_gradients = self._get_masked_gradients(gradients, noise[0])
            self.optimization_meta["top_k_mask"] = tf.reshape(masked_gradients, tf.shape(noise[0]))
        masked_gradients = gradients * self.optimization_meta["top_k_mask"]
        optimizer.apply(noise, masked_gradients)

    def _optimize_incremental(self,
                              noise: tf.Variable,
                              gradients: tf.Tensor,
                              ) -> None:
        masked_gradients = self._get_masked_gradients(gradients, noise[0])
        attacked_pixels = self._get_attacked_pixels(masked_gradients, noise[0])
        noise.assign(tf.where(attacked_pixels != 0, attacked_pixels, noise))

    def _get_masked_gradients(self, 
                        gradients: tf.Tensor,
                        tensor: tf.Variable) -> tf.Tensor:
        flat_gradients = tf.reshape(gradients, [-1])

        if self.pixel_selector == "positive":
            flat_scores = flat_gradients
        elif self.pixel_selector == "negative":
            flat_scores = -flat_gradients
        elif self.pixel_selector == "absolute":
            flat_scores = tf.abs(flat_gradients)
        else:
            raise ValueError(f"Invalid direction: {self.pixel_selector}")

        _, top_k_indices = tf.math.top_k(flat_scores, k=self.increment_per_iteration)

        top_k_mask = tf.scatter_nd(
            indices=tf.expand_dims(top_k_indices, axis=1),
            updates=tf.ones_like(top_k_indices, dtype=tf.float32),
            shape=tf.shape(flat_gradients),
        )

        self.optimization_meta["top_k_mask"] = tf.reshape(top_k_mask, tf.shape(tensor))

        masked_gradients = flat_gradients * top_k_mask

        return masked_gradients
    
    def _get_attacked_pixels(self, masked_gradients: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
        if self.direction == "positive":
            updated_gradients = tf.where(
                tf.not_equal(masked_gradients, 0),
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[1], dtype=masked_gradients.dtype)),
                masked_gradients
            )
        elif self.direction == "negative":
            updated_gradients = tf.where(
                tf.not_equal(masked_gradients, 0),
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[0], dtype=masked_gradients.dtype)),
                masked_gradients
            )
        elif self.direction == "bi":
            updated_gradients = tf.where(
                masked_gradients > 0,
                tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[0], dtype=masked_gradients.dtype)),
                tf.where(
                    masked_gradients < 0,
                    tf.fill(tf.shape(masked_gradients), tf.cast(self.epsilon[1], dtype=masked_gradients.dtype)),
                    masked_gradients
                )
            )
        else:
            raise ValueError(f"Invalid direction: {self.direction}")
        
        return tf.reshape(updated_gradients, tf.shape(tensor))
    
    def _reset_optimization_meta(self):
        self.optimization_meta = {
            "iterative_mask": None,
        }