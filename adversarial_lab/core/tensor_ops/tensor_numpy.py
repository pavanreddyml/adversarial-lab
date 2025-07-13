from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import gaussian_filter

from typing import Union, List, Any

class TensorOpsNumpy:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = NumpyLosses()
        self.optimizers = NumpyOptimizers()

    @staticmethod
    def tensor(arr: Union[np.ndarray, List[float], List[int], float, int]) -> np.ndarray:
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def numpy(array: Union[np.ndarray, List[float], List[int], float, int]) -> np.ndarray:
        return np.array(array)

    @staticmethod
    def constant(value: Union[float, int], dtype: Any = np.float32) -> np.ndarray:
        return np.array(value, dtype=dtype)

    @staticmethod
    def variable(arr: Union[np.ndarray, List[float], List[int], float, int]) -> np.ndarray:
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def assign(array: np.ndarray, value: Union[np.ndarray, List[float], List[int], float, int]) -> np.ndarray:
        array[:] = np.array(value)
        return array

    @staticmethod
    def cast(array: np.ndarray, dtype: Any) -> np.ndarray:
        return array.astype(dtype)

    @staticmethod
    def has_batch_dim(array: np.ndarray, axis: int = 0) -> bool:
        return array.ndim > axis and array.shape[axis] == 1

    @staticmethod
    def add_batch_dim(array: np.ndarray, axis: int = 0) -> np.ndarray:
        if TensorOpsNumpy.has_batch_dim(array, axis):
            return array
        return np.expand_dims(array, axis=axis)

    @staticmethod
    def is_zero(array: np.ndarray) -> bool:
        return np.all(array == 0)

    @staticmethod
    def remove_batch_dim(array: np.ndarray, axis: int = 0) -> np.ndarray:
        if TensorOpsNumpy.has_batch_dim(array, axis):
            return np.squeeze(array, axis=axis)
        return array

    @staticmethod
    def zeros_like(array: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
        return np.zeros_like(array, dtype=dtype)

    @staticmethod
    def ones_like(array: np.ndarray) -> np.ndarray:
        return np.ones_like(array)

    @staticmethod
    def abs(array: np.ndarray) -> np.ndarray:
        return np.abs(array)

    @staticmethod
    def norm(array: np.ndarray, p: float = 2.0) -> float:
        return np.linalg.norm(array, ord=p)

    @staticmethod
    def sub(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
        return np.subtract(array_a, array_b)

    @staticmethod
    def add(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
        return np.add(array_a, array_b)

    @staticmethod
    def min(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.min(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.max(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def clip(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(array, min_val, max_val)

    @staticmethod
    def reduce_max(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.max(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_min(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.min(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_mean(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_sum(array: np.ndarray, axis: Any = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(array, axis=axis, keepdims=keepdims)

    @staticmethod
    def reduce_all(array: np.ndarray) -> bool:
        return np.all(array)

    @staticmethod
    def random_uniform(shape: List[int], minval: float = 0.0, maxval: float = 1.0) -> np.ndarray:
        return np.random.uniform(low=minval, high=maxval, size=shape).astype(np.float32)

    @staticmethod
    def random_normal(shape: List[int]) -> np.ndarray:
        return np.random.normal(size=shape).astype(np.float32)

    @staticmethod
    def relu(array: np.ndarray) -> np.ndarray:
        return np.maximum(array, 0)

    @staticmethod
    def tensordot(a: np.ndarray, b: np.ndarray, axes: Union[int, List[int]]) -> np.ndarray:
        return np.tensordot(a, b, axes=axes)

    @staticmethod
    def reshape(array: np.ndarray, shape: List[int]) -> np.ndarray:
        return np.reshape(array, shape)
    
    @staticmethod
    def gaussian_blur(array: np.ndarray, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
        """Apply a Gaussian blur to a 2D or 3D NumPy array."""
        from scipy.ndimage import gaussian_filter

        if array.ndim == 2:
            return gaussian_filter(array, sigma=sigma)

        if array.ndim == 3:
            # Apply per-channel
            return np.stack([gaussian_filter(array[..., c], sigma=sigma) for c in range(array.shape[-1])], axis=-1)

        if array.ndim == 4:
            # Batch processing: [B, H, W, C]
            return np.stack([
                np.stack([gaussian_filter(image[..., c], sigma=sigma) for c in range(image.shape[-1])], axis=-1)
                for image in array
            ], axis=0)

        raise ValueError(f"Unsupported array shape for gaussian_blur: {array.shape}")


class NumpyLosses:
    def __init__(self):
        pass

    @staticmethod
    def binary_crossentropy(target: np.ndarray,
                            predictions: np.ndarray,
                            logits: np.ndarray = None,
                            from_logits: bool = False) -> float:
        preds = logits if from_logits and logits is not None else predictions
        preds = 1 / (1 + np.exp(-preds)) if from_logits else preds
        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        loss = -(target * np.log(preds) + (1 - target) * np.log(1 - preds))
        return np.mean(loss)

    @staticmethod
    def categorical_crossentropy(target: np.ndarray,
                                 predictions: np.ndarray,
                                 logits: np.ndarray = None,
                                 from_logits: bool = False) -> float:
        preds = logits if from_logits and logits is not None else predictions
        if from_logits:
            e_x = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
            preds = e_x / np.sum(e_x, axis=-1, keepdims=True)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        loss = -np.sum(target * np.log(preds), axis=-1)
        return np.mean(loss)

    @staticmethod
    def mean_absolute_error(target: np.ndarray,
                            predictions: np.ndarray,
                            logits: np.ndarray = None,
                            from_logits: bool = False) -> float:
        return np.mean(np.abs(target - predictions))

    @staticmethod
    def mean_squared_error(target: np.ndarray,
                           predictions: np.ndarray,
                           logits: np.ndarray = None,
                           from_logits: bool = False) -> float:
        return np.mean(np.square(target - predictions))


class NumpyOptimizers:
    def __init__(self):
        pass

    class NumpyOptimizer(ABC):
        @abstractmethod
        def apply(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
            pass

    class SGD(NumpyOptimizer):
        def __init__(self, learning_rate: float = 0.01):
            self.learning_rate = learning_rate

        def apply(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
            for p, g in zip(params, grads):
                p -= self.learning_rate * g
            return params

    class Adam(NumpyOptimizer):
        def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
            self.learning_rate = learning_rate
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon
            self.m = {}
            self.v = {}
            self.t = 0

        def apply(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
            self.t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                if i not in self.m:
                    self.m[i] = np.zeros_like(g)
                    self.v[i] = np.zeros_like(g)
                self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * g
                self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (g ** 2)

                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

                p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            return params

    class PGD(NumpyOptimizer):
        def __init__(self, learning_rate: float = 0.01, projection_fn: Any = None, topk_percent: float = 1.0):
            self.learning_rate = learning_rate
            self.projection_fn = projection_fn
            self.topk_percent = topk_percent

        def apply(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
            for p, g in zip(params, grads):
                if self.topk_percent < 1.0:
                    flat = g.flatten()
                    k = int(self.topk_percent * flat.size)
                    if k > 0:
                        idx = np.argpartition(np.abs(flat), -k)[-k:]
                        mask = np.zeros_like(flat)
                        mask[idx] = 1
                        masked_grad = (flat * mask).reshape(g.shape)
                    else:
                        masked_grad = np.zeros_like(g)
                else:
                    masked_grad = g

                p -= self.learning_rate * np.sign(masked_grad)

                if self.projection_fn is not None:
                    p[:] = self.projection_fn(p)

            return params
        
    @staticmethod
    def apply(optimizer: NumpyOptimizer, weights, gradients):
        return optimizer.apply(weights, gradients)
