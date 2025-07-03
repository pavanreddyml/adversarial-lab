import numpy as np

from typing import Union, TYPE_CHECKING, Callable


if TYPE_CHECKING:
    try:
        import tensorflow as tf
        from adversarial_lab.core.tensor_ops.tensor_tf import TensorOpsTF
    except ImportError:
        tf = None

    try:
        import torch
        from adversarial_lab.core.tensor_ops.tensor_torch import TensorOpsTorch
    except ImportError:
        torch = None
        
    
    from adversarial_lab.core.tensor_ops.tensor_numpy import TensorOpsNumpy

TensorType = Union["tf.Tensor", "torch.Tensor", "np.ndarray"]
TensorVariableType = Union["tf.Variable", "torch.Tensor", "np.ndarray"]
LossType = Union["tf.keras.losses.Loss", "torch.nn.modules.loss._Loss", TensorType, "np.ndarray", float]
OptimizerType = Union["tf.optimizers.Optimizer", "torch.optim.Optimizer"]
ModelType = Union["tf.keras.Model", "torch.nn.Module", "tf.keras.Sequential", Callable]
TensorOpsType = Union["TensorOpsTorch", "TensorOpsTF", "TensorOpsNumpy"]

