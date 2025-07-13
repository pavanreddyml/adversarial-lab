import tensorflow as tf
import numpy as np

from typing import Union, List, Any, Callable
from adversarial_lab.core.types import TensorType, TensorVariableType, LossType, OptimizerType


class TensorOpsTF:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = TFLosses()
        self.optimizers = TFOptimizers()

    @staticmethod
    def tensor(arr: Union[np.ndarray, List[float], List[int], TensorType]) -> TensorType:
        """Convert numpy array or list to a TensorFlow tensor."""
        return tf.convert_to_tensor(arr, dtype=tf.float32)
    
    @staticmethod
    def numpy(tensor: TensorType) -> np.ndarray:
        """Convert a TensorFlow tensor to a numpy array."""
        return tensor.numpy() if isinstance(tensor, tf.Tensor) else np.array(tensor)
    
    @staticmethod
    def constant(value: Union[float, int], dtype: Any) -> TensorType:
        """Create a TensorFlow constant."""
        return tf.constant(value, dtype=dtype)

    @staticmethod
    def variable(arr: Union[np.ndarray, List[float], List[int], TensorType]) -> TensorVariableType:
        """Convert numpy array or list to a TensorFlow variable."""
        return tf.Variable(tf.convert_to_tensor(arr, dtype=tf.float32))

    @staticmethod
    def assign(tensor: TensorVariableType, value: Union[np.ndarray, List[float], List[int], TensorType]) -> None:
        """Assign a new value to a TensorFlow variable."""
        tensor.assign(tf.convert_to_tensor(value))

    @staticmethod
    def cast(tensor: TensorType, dtype: Any) -> TensorType:
        """Cast tensor to a specified data type."""
        return tf.cast(tensor, dtype)
    
    @staticmethod
    def has_batch_dim(tensor: TensorType, axis: int = 0) -> bool:
        """Check if the tensor has a batch dimension at the specified axis."""
        return tensor.shape.rank is not None and tensor.shape.rank > axis and tensor.shape[axis] == 1
    
    @staticmethod
    def add_batch_dim(tensor: TensorType, axis: int = 0) -> TensorType:
        """Add a batch dimension to the tensor if it doesn't already exist."""
        if tensor.shape.rank is not None and tensor.shape.rank > axis and tensor.shape[axis] == 1:
            return tensor
        return tf.expand_dims(tensor, axis=axis)
    
    @staticmethod
    def is_zero(tensor: TensorType) -> bool:
        """Check if all elements in the tensor are zero."""
        return tf.reduce_all(tensor == 0)
    
    @staticmethod
    def remove_batch_dim(tensor: TensorType, axis: int = 0) -> TensorType:
        """Remove a batch dimension from the tensor if it exists."""
        if tensor.shape.rank is not None and tensor.shape.rank > axis and tensor.shape[axis] == 1:
            return tf.squeeze(tensor, axis=axis)
        return tensor

    @staticmethod
    def zeros_like(tensor: TensorType, dtype: Any = "float32") -> TensorType:
        """Create a tensor of ones with the same shape as the input tensor."""
        return tf.zeros_like(tensor, dtype=dtype)

    @staticmethod
    def ones_like(tensor: TensorType) -> TensorType:
        """Create a tensor of ones with the same shape as the input tensor."""
        return tf.ones_like(tensor)

    @staticmethod
    def abs(tensor: TensorType) -> TensorType:
        """Return absolute values of elements in the tensor."""
        return tf.abs(tensor)

    @staticmethod
    def norm(tensor: TensorType, p: float = 2) -> TensorType:
        """Compute the Lp norm of the tensor."""
        return tf.reduce_sum(tf.abs(tensor) ** p) ** (1.0 / p)
    
    @staticmethod
    def sub(tensor_a: TensorType, tensor_b: TensorType) -> TensorType:
        """Subtract two tensors."""
        return tf.subtract(tensor_a, tensor_b)
    
    @staticmethod
    def add(tensor_a: TensorType, tensor_b: TensorType) -> TensorType:
        """Add two tensors."""
        return tf.add(tensor_a, tensor_b)
    
    @staticmethod
    def min(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the minimum value in the tensor."""
        return tf.reduce_min(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def max(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the maximum value in the tensor."""
        return tf.reduce_max(input_tensor=tensor, axis=axis, keepdims=keepdims)

    @staticmethod
    def clip(tensor: TensorVariableType, min_val: float, max_val: float) -> TensorType:
        """Clip tensor values between min and max."""
        return tf.clip_by_value(tensor, min_val, max_val)
    
    @staticmethod
    def reduce_max(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the maximum value in the tensor."""
        return tf.reduce_max(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_min(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the minimum value in the tensor."""
        return tf.reduce_min(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_mean(tensor: TensorType, axis: Any | None = None, keepdims: bool = False) -> TensorType:
        """Compute the mean value in the tensor."""
        return tf.reduce_mean(input_tensor=tensor, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def reduce_sum(tensor: TensorType, axis: Any | None = None, keepdims: bool = False, name: Any | None = None) -> TensorType:
        """Compute the sum of all elements in the tensor."""
        return tf.reduce_sum(input_tensor=tensor, axis=axis, keepdims=keepdims, name=name)
    
    @staticmethod
    def reduce_all(tensor: TensorType) -> bool:
        """Check if all elements in the tensor are True."""
        return tf.reduce_all(tensor)
    
    @staticmethod
    def random_uniform(shape: List[int], minval: float, maxval: float) -> TensorType:
        """Generate a tensor with random uniform values."""
        return tf.random.uniform(shape, minval=minval, maxval=maxval)
    
    @staticmethod
    def random_normal(shape: List[int]) -> TensorType:
        """Generate a tensor with random normal values."""
        return tf.random.normal(shape)
    
    @staticmethod
    def relu(tensor: TensorType) -> TensorType:
        """Compute the ReLU activation function."""
        return tf.nn.relu(tensor)
    
    @staticmethod
    def tensordot(a: TensorType, b: TensorType, axes: Union[int, List[int]]) -> TensorType:
        """Compute the tensor dot product."""
        return tf.tensordot(a, b, axes=axes)
    
    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> TensorType:
        """Reshape the tensor to the specified shape."""
        return tf.reshape(tensor, shape)
    
    @staticmethod
    def gaussian_blur(tensor: TensorType, sigma: float = 1.0, kernel_size: int = 5) -> TensorType:
        """Apply a Gaussian blur to a 4D image tensor using depthwise convolution."""
        if tensor.ndim == 3:
            tensor = tf.expand_dims(tensor, axis=0)  # Make it 4D

        def get_gaussian_kernel(size: int, sigma: float) -> TensorType:
            coords = tf.range(size, dtype=tf.float32) - (size - 1.0) / 2.0
            g = tf.exp(-0.5 * tf.square(coords) / tf.square(sigma))
            g /= tf.reduce_sum(g)
            return g

        kernel_1d = get_gaussian_kernel(kernel_size, sigma)
        kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
        kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]
        channels = tf.shape(tensor)[-1]
        kernel = tf.repeat(kernel_2d, channels, axis=2)

        blurred = tf.nn.depthwise_conv2d(tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(blurred, axis=0) if tensor.shape[0] == 1 else blurred
    


class TFLosses:
    def __init__(self):
        pass

    @staticmethod
    def binary_crossentropy(target: TensorType,
                            predictions: TensorType,
                            logits: TensorType,
                            from_logits: bool) -> TensorType:
        """Compute binary cross-entropy loss."""
        preds = logits if from_logits else predictions
        loss = tf.keras.losses.binary_crossentropy(
            target, preds, from_logits=from_logits)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def categorical_crossentropy(target: TensorType,
                                 predictions: TensorType,
                                 logits: TensorType,
                                 from_logits: bool) -> TensorType:
        """Compute Categorical cross-entropy loss."""
        preds = logits if from_logits else predictions
        loss = tf.keras.losses.categorical_crossentropy(
            target, preds, from_logits=from_logits)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def mean_absolute_error(target: TensorType,
                            predictions: TensorType
                            ) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss = loss_fn(target, predictions)
        return tf.reduce_mean(loss)

    @staticmethod
    def mean_squared_error(target: TensorType,
                           predictions: TensorType
                           ) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(target, predictions)
        return tf.reduce_mean(loss)


class TFOptimizers:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def Adam(learning_rate: float = 0.001,
             beta_1: float = 0.9,
             beta_2: float = 0.999,
             epsilon: float = 1e-8
             ) -> OptimizerType:
        """Create an Adam optimizer."""
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon
        )
    
    @staticmethod
    def SGD(learning_rate: float = 0.01,
            momentum: float = 0.0,
            nesterov: bool = False,
            weight_decay: float = None,
            ) -> OptimizerType:
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    
    @staticmethod
    def PGD(
        learning_rate: float = 0.01,
        projection_fn: Callable[[TensorType], TensorType] = None,
        topk_percent: float = 1.0  # 1.0 = full gradient
    ) -> OptimizerType:

        class PGDOptimizer(tf.keras.optimizers.Optimizer):
            def __init__(self, learning_rate=0.01, projection_fn=None, topk_percent=1.0, name="PGD", **kwargs):
                super().__init__(name=name, learning_rate=learning_rate, **kwargs)
                self.projection_fn = projection_fn
                self.topk_percent = topk_percent

            def update_step(self, gradient, variable, learning_rate):
                if self.topk_percent < 1.0:
                    k = tf.cast(tf.size(gradient) * self.topk_percent, tf.int32)
                    flat_grad = tf.reshape(gradient, [-1])
                    _, topk_indices = tf.math.top_k(tf.abs(flat_grad), k=k, sorted=False)
                    mask = tf.scatter_nd(
                        indices=tf.expand_dims(topk_indices, 1),
                        updates=tf.ones_like(topk_indices, dtype=gradient.dtype),
                        shape=tf.shape(flat_grad)
                    )
                    masked_grad = tf.reshape(flat_grad * mask, tf.shape(gradient))
                else:
                    masked_grad = gradient

                variable.assign_sub(learning_rate * tf.sign(masked_grad))

                if self.projection_fn:
                    variable.assign(self.projection_fn(variable))

            def get_config(self):
                config = super().get_config()
                config.update({
                    "learning_rate": float(tf.keras.backend.get_value(self._serialize_hyperparameter("learning_rate"))),
                    "projection_fn": self.projection_fn,
                    "topk_percent": self.topk_percent,
                })
                return config

        return PGDOptimizer(learning_rate=learning_rate, projection_fn=projection_fn, topk_percent=topk_percent)
    
    @staticmethod
    def apply(optimizer: tf.keras.optimizers.Optimizer,
              variable_tensor: List[TensorVariableType],
              gradients: List[TensorType]) -> None:
        """Apply gradients to update model weights."""
        optimizer.apply_gradients(zip(gradients, variable_tensor))
    
    @staticmethod
    def has_param(optimizer: tf.keras.optimizers.Optimizer, 
                  param_name: str
                  ) -> bool:
        return param_name in optimizer._hyper or hasattr(optimizer, param_name)

    @staticmethod
    def update_param(optimizer: tf.keras.optimizers.Optimizer, 
                     param_name: str, 
                     value
                     ) -> None:
        if param_name in optimizer._hyper:
            optimizer._hyper[param_name] = value
        elif hasattr(optimizer, param_name):
            setattr(optimizer, param_name, value)
        else:
            raise ValueError(f"Parameter '{param_name}' not found in optimizer.")