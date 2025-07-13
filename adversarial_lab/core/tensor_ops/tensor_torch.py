import torch
import torch.nn.functional as F
from typing import List, Callable, Any, Union
import numpy as np

from adversarial_lab.core.types import TensorType, TensorVariableType, LossType, OptimizerType

class TensorOpsTorch:
    def __init__(self, *args, **kwargs) -> None:
        self.losses = TorchLosses()
        self.optimizers = TorchOptimizers()

    @staticmethod
    def tensor(arr: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or list to a PyTorch tensor."""
        return torch.tensor(arr, dtype=torch.float32)
    
    @staticmethod
    def numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to a numpy array."""
        return tensor.detach().cpu().numpy() if tensor.is_cuda else tensor.numpy()
    
    @staticmethod
    def constant(value: Union[float, int], dtype: Any) -> torch.Tensor:
        """Create a PyTorch constant."""
        return torch.tensor(value, dtype=dtype)

    @staticmethod
    def variable(arr: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> torch.Tensor:
        """Convert numpy array or list to a PyTorch tensor with gradients enabled."""
        return torch.tensor(arr, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def assign(tensor: torch.Tensor, value: Union[np.ndarray, List[float], List[int], torch.Tensor]) -> None:
        """Assign a new value to a PyTorch tensor (in-place update)."""
        with torch.no_grad():
            tensor.copy_(torch.tensor(value, dtype=torch.float32))

    @staticmethod
    def cast(tensor: torch.Tensor, dtype: Any) -> torch.Tensor:
        """Cast tensor to a specified data type."""
        return tensor.to(dtype)
    
    @staticmethod
    def has_batch_dim(tensor: torch.Tensor, axis: int = 0) -> bool:
        if tensor.dim() > axis and tensor.size(axis) == 1:
            return True
    
    @staticmethod
    def add_batch_dim(tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
        if tensor.dim() > axis and tensor.size(axis) == 1:
            return tensor
        return tensor.unsqueeze(dim=axis)

    @staticmethod
    def remove_batch_dim(tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
        if tensor.dim() > axis and tensor.size(axis) == 1:
            return tensor.squeeze(dim=axis)
        return tensor
    
    @staticmethod
    def is_zero(tensor: torch.Tensor) -> bool:
        """Check if all elements in the tensor are zero."""
        return torch.all(tensor == 0)

    @staticmethod
    def zeros_like(tensor: torch.Tensor) -> torch.Tensor:
        """Create a tensor of ones with the same shape as the input tensor."""
        return torch.zeros_like(tensor)

    @staticmethod
    def ones_like(tensor: torch.Tensor) -> torch.Tensor:
        """Create a tensor of ones with the same shape as the input tensor."""
        return torch.ones_like(tensor)

    @staticmethod
    def abs(tensor: torch.Tensor) -> torch.Tensor:
        """Return absolute values of elements in the tensor."""
        return torch.abs(tensor)

    @staticmethod
    def norm(tensor: torch.Tensor, p: float) -> torch.Tensor:
        """Compute the Lp norm of the tensor."""
        return torch.norm(tensor, p=p)
    
    @staticmethod
    def sub(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        return torch.sub(tensor_a, tensor_b)

    @staticmethod
    def add(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        return torch.add(tensor_a, tensor_b)

    @staticmethod
    def min(tensor: torch.Tensor, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.min(tensor)
        return torch.min(tensor, dim=axis, keepdim=keepdims).values

    @staticmethod
    def max(tensor: torch.Tensor, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.max(tensor)
        return torch.max(tensor, dim=axis, keepdim=keepdims).values

    @staticmethod
    def clip(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Clip tensor values between min and max."""
        return torch.clamp(tensor, min=min_val, max=max_val)

    @staticmethod
    def reduce_max(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the maximum value in the tensor."""
        return torch.max(tensor, dim=axis, keepdim=keepdims)[0] if axis is not None else torch.max(tensor)

    @staticmethod
    def reduce_min(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the minimum value in the tensor."""
        return torch.min(tensor, dim=axis, keepdim=keepdims)[0] if axis is not None else torch.min(tensor)

    @staticmethod
    def reduce_mean(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the mean value in the tensor."""
        return torch.mean(tensor, dim=axis, keepdim=keepdims) if axis is not None else torch.mean(tensor)

    @staticmethod
    def reduce_sum(tensor: torch.Tensor, axis: Any | None = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the sum of all elements in the tensor."""
        return torch.sum(tensor, dim=axis, keepdim=keepdims) if axis is not None else torch.sum(tensor)
    
    @staticmethod
    def reduce_all(tensor: torch.Tensor) -> bool:
        """Compute the logical AND of all elements in the tensor."""
        return torch.all(tensor)

    @staticmethod
    def random_normal(shape: List[int]) -> torch.Tensor:
        """Generate a tensor with random normal values."""
        return torch.randn(shape)

    @staticmethod
    def random_uniform(shape: List[int], minval: float, maxval: float) -> torch.Tensor:
        """Generate a tensor with random uniform values."""
        return torch.empty(shape).uniform_(minval, maxval)
    
    @staticmethod
    def relu(tensor: torch.Tensor) -> torch.Tensor:
        """Compute the ReLU activation function."""
        return F.relu(tensor)
    
    @staticmethod
    def tensordot(a: TensorType, b: TensorType, axes: Union[int, List[int]]) -> TensorType:
        """Compute the tensor dot product."""
        return torch.tensordot(a, b, dims=axes)
    
    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> TensorType:
        """Reshape the tensor to the specified shape."""
        return tensor.view(*shape)
    
    @staticmethod
    def gaussian_blur(tensor: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
        """Apply a Gaussian blur to a 4D or 3D tensor using depthwise convolution."""

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension

        channels = tensor.size(1) if tensor.dim() == 4 else tensor.size(0)
        
        def get_gaussian_kernel(size, sigma):
            coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
            kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            return kernel

        kernel_1d = get_gaussian_kernel(kernel_size, sigma)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).to(tensor.device)

        # Apply depthwise convolution
        padding = kernel_size // 2
        blurred = F.conv2d(tensor, kernel_2d, groups=channels, padding=padding)

        return blurred.squeeze(0) if blurred.size(0) == 1 else blurred


class TorchLosses:
    def __init__(self):
        pass

    @staticmethod
    def binary_crossentropy(target: TensorType,
                            predictions: TensorType,
                            logits: TensorType,
                            from_logits: bool) -> TensorType:
        """Compute Binary Cross-Entropy (BCE) loss."""
        preds = logits if from_logits else predictions
        loss = F.binary_cross_entropy_with_logits(preds, target) if from_logits else F.binary_cross_entropy(preds, target)
        return loss.mean()

    @staticmethod
    def categorical_crossentropy(target: TensorType,
                                 predictions: TensorType,
                                 logits: TensorType,
                                 from_logits: bool) -> TensorType:
        """Compute Categorical Cross-Entropy (CCE) loss."""
        preds = logits if from_logits else predictions
        loss = F.cross_entropy(preds, target) if from_logits else -(target * preds.log()).sum(dim=1).mean()
        return loss

    @staticmethod
    def mean_absolute_error(target: TensorType,
                            predictions: TensorType) -> TensorType:
        """Compute Mean Absolute Error (MAE)."""
        loss = F.l1_loss(predictions, target, reduction='mean')
        return loss

    @staticmethod
    def mean_squared_error(target: TensorType,
                           predictions: TensorType) -> TensorType:
        """Compute Mean Squared Error (MSE)."""
        loss = F.mse_loss(predictions, target, reduction='mean')
        return loss
    

class TorchOptimizers:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def Adam(learning_rate: float = 0.001,
             beta_1: float = 0.9,
             beta_2: float = 0.999,
             epsilon: float = 1e-8
             ) -> OptimizerType:
        """Apply Adam optimizer."""
        import torch.optim as optim
        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = optim.Adam([temp_param], lr=learning_rate, betas=(beta_1, beta_2), eps=epsilon)
        opt.param_groups[0]["params"] = [] 
        return opt
    
    @staticmethod
    def SGD(learning_rate: float = 0.01,
            momentum: float = 0.0,
            nesterov: bool = False,
            weight_decay: float = 0.0,
            ) -> OptimizerType:
        """Apply SGD optimizer."""
        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = torch.optim.SGD(
            [temp_param],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
            nesterov=nesterov
        )
        opt.param_groups[0]["params"] = [] 
        return opt

    @staticmethod
    def PGD(learning_rate: float = 0.01,
            projection_fn: Callable[[TensorType], TensorType] = None,
            topk_percent: float = 1.0
            ) -> OptimizerType:

        class PGDOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01, projection_fn=None, topk_percent=1.0):
                defaults = dict(lr=lr, projection_fn=projection_fn, topk_percent=topk_percent)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                for group in self.param_groups:
                    lr = group['lr']
                    projection_fn = group['projection_fn']
                    topk_percent = group['topk_percent']

                    for param in group['params']:
                        if param.grad is None:
                            continue

                        grad = param.grad

                        if topk_percent < 1.0:
                            k = int(topk_percent * grad.numel())
                            if k > 0:
                                flat_grad = grad.view(-1)
                                _, topk_indices = torch.topk(flat_grad.abs(), k, sorted=False)
                                mask = torch.zeros_like(flat_grad)
                                mask[topk_indices] = 1.0
                                masked_grad = (flat_grad * mask).view_as(grad)
                            else:
                                masked_grad = torch.zeros_like(grad)
                        else:
                            masked_grad = grad

                        param.data -= lr * masked_grad

                        if projection_fn:
                            param.data = projection_fn(param.data)

                if closure is not None:
                    return closure()

        temp_param = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        opt = PGDOptimizer([temp_param], lr=learning_rate, projection_fn=projection_fn, topk_percent=topk_percent)
        opt.param_groups[0]["params"] = []
        return opt

    @staticmethod
    def apply(optimizer: OptimizerType,
              variable_tensor: List[TensorType],
              gradients: List[TensorType]) -> None:
        """Apply gradients to update model weights."""
        if len(optimizer.param_groups[0]['params']) == 0:
            optimizer.add_param_group({'params': variable_tensor})

        for param, grad in zip(variable_tensor, gradients):
            param.grad = grad

        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def has_param(optimizer: torch.optim.Optimizer, 
                  param_name: str
                  ) -> bool:
        return any(param_name in group for group in optimizer.param_groups)


    @staticmethod
    def update_param(optimizer: torch.optim.Optimizer, 
                     param_name: str, 
                     value
                     ) -> None:
        updated = False
        for group in optimizer.param_groups:
            if param_name in group:
                group[param_name] = value
                updated = True
        if not updated:
            raise ValueError(f"Parameter '{param_name}' not found in optimizer.")