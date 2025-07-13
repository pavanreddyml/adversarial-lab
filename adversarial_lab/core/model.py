from abc import ABC, ABCMeta, abstractmethod

import importlib
import numpy as np
from copy import deepcopy

from adversarial_lab.core.losses import Loss
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.exceptions import IndifferentiabilityError, VectorDimensionsError
from adversarial_lab.core.gradient_estimator import GradientEstimator, DummyGradientEstimator

from typing import Dict, Any, List, Tuple, Callable, TYPE_CHECKING, Optional, List, Union
from adversarial_lab.core.types import TensorType, TensorVariableType, ModelType, LossType


tf = None
torch = None

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class ALModelMeta(ABCMeta):
    def __call__(cls, model, *args, **kwargs):
        global torch, tf
        if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
            framework = "torch"
        elif hasattr(model, "layers") and hasattr(model, "count_params"):
            framework = "tf"
        elif callable(model):
            framework = "numpy"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        if framework == "torch":
            torch = importlib.import_module("torch")
            specific_class = ALModelTorch
        elif framework == "tf":
            tf = importlib.import_module("tensorflow")
            specific_class = ALModelTF
        elif framework == "numpy":
            tf = importlib.import_module("numpy")
            specific_class = ALModelNumpy

        return specific_class(model, *args, **kwargs)


class ALModelBase(ABC):
    def __init__(self,
                 model: ModelType,
                 framework: str,
                 efficient_mode: Optional[int] = None,
                 efficient_mode_indexes: Optional[List[int]] = None,
                 gradient_estimator: Optional[GradientEstimator] = None) -> None:
        self.model = deepcopy(model)
        self.model_info = self.get_info(self.model)
        self.framework = framework
        self._compute_jacobian = False

        if efficient_mode is None:
            self.efficient_mode = None
        elif not isinstance(efficient_mode, int):
            raise ValueError(
                f"Efficient mode must be an integer, got {type(efficient_mode)} instead.")
        else:
            self.efficient_mode = efficient_mode

        if not isinstance(efficient_mode_indexes, (list, type(None))):
            raise ValueError(
                f"Efficient mode indexes must be a list or None, got {type(efficient_mode_indexes)} instead.")
        if efficient_mode_indexes is not None and not all(isinstance(i, int) for i in efficient_mode_indexes):
            raise ValueError(
                "All elements in efficient_mode_indexes must be integers.")
        self.efficient_mode_indexes = efficient_mode_indexes or []

        if gradient_estimator is None:
            self.gradient_estimator = DummyGradientEstimator()
        elif not isinstance(gradient_estimator, GradientEstimator):
            raise TypeError(
                f"gradient_estimator must be an instance of GradientEstimator, got {type(gradient_estimator)} instead.")
        else:
            self.gradient_estimator = gradient_estimator

    @abstractmethod
    def get_info(self, model) -> Dict[str, Any]:
        pass

    @abstractmethod
    def calculate_gradients(self,
                            sample: TensorType,
                            noise: List[TensorVariableType],
                            apply_noise_fn: Callable,
                            target_vector: TensorType,
                            loss: LossType,
                            preprocess_fn: Optional[Callable] = None
                            ) -> Tuple[TensorType, TensorType]:
        pass

    @abstractmethod
    def predict(self, x: TensorType | np.ndarray, preprocess_fn: Optional[Callable] = None) -> Any:
        pass

    @abstractmethod
    def forward(self, x: TensorType | np.ndarray) -> Any:
        pass

    def set_compute_jacobian(self,
                             compute_jacobian: bool
                             ) -> None:
        self._compute_jacobian = compute_jacobian

    def set_max_queries(self, max_queries: int) -> None:
        pass

    def reset_query_count(self) -> None:
        pass


class ALModelTorch(ALModelBase):
    def __init__(self,
                 model: ModelType,
                 efficient_mode: Optional[int] = None,
                 efficient_mode_indexes: Optional[List[int]] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model=model, 
                         framework="torch", 
                         efficient_mode=efficient_mode, 
                         efficient_mode_indexes=efficient_mode_indexes,
                         gradient_estimator=None)

    def get_info(self,
                 model: ModelType
                 ) -> Dict[str, Any]:
        num_params = sum(p.numel() for p in model.parameters())

        layers_info = []
        total_units = 0
        for layer in model.children():
            layer_info = {
                "layer_name": layer.__class__.__name__,
                "layer_type": type(layer).__name__,
            }

            if hasattr(layer, 'weight'):
                layer_info["units"] = layer.weight.shape[0]
                total_units += layer.weight.shape[0]

            if isinstance(layer, torch.nn.ReLU):
                layer_info["activation"] = "ReLU"
            elif isinstance(layer, torch.nn.Sigmoid):
                layer_info["activation"] = "Sigmoid"
            elif isinstance(layer, torch.nn.Tanh):
                layer_info["activation"] = "Tanh"
            elif isinstance(layer, torch.nn.LeakyReLU):
                layer_info["activation"] = "LeakyReLU"
            elif isinstance(layer, torch.nn.Softmax):
                layer_info["activation"] = "Softmax"

            layers_info.append(layer_info)

        input_shape = None
        output_shape = None
        try:
            example_input = torch.randn(1, *model.input_shape)
            output = model(example_input)
            input_shape = example_input.shape
            output_shape = output.shape
        except AttributeError:
            pass

        param_info = [
            {"name": name, "shape": param.shape,
                "requires_grad": param.requires_grad}
            for name, param in model.named_parameters()
        ]

        model_info = {
            "num_params": num_params,
            "total_units": total_units,
            "layers_info": layers_info,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "params": param_info,
        }
        return model_info

    def calculate_gradients(self,
                            sample: TensorType,
                            noise: List[TensorVariableType],
                            apply_noise_fn: Callable,
                            target_vector: TensorType,
                            loss: LossType,
                            preprocess_fn: Optional[Callable] = None
                            ) -> Tuple[TensorType, TensorType]:
        raise NotImplementedError()

    def predict(self,
                x: TensorType | np.ndarray,
                preprocess_fn: Optional[Callable] = None
                ) -> TensorType:
        raise NotImplementedError()

    def forward(self,
                x: TensorType | np.ndarray
                ) -> TensorType:
        raise NotImplementedError()
    
    


class ALModelTF(ALModelBase):
    def __init__(self,
                 model: ModelType,
                 efficient_mode: Optional[int] = None,
                 efficient_mode_indexes: Optional[List[int]] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 *args,
                 **kwargs) -> None:       
        super().__init__(
            model=model,
            framework="tf",
            efficient_mode=efficient_mode,
            efficient_mode_indexes=efficient_mode_indexes,
            gradient_estimator=None
        )
        self.act = getattr(self.model.layers[-1], "activation", None)
        self.model.layers[-1].activation = None
        self.tensor_ops = TensorOps(framework="tf")

        if self.act is None:
            self.act = lambda x: x

    def get_info(self,
                 model: ModelType) -> Dict[str, Any]:
        num_params = model.count_params()

        layers_info = []
        total_units = 0
        for layer in model.layers:
            layer_info = {
                "layer_name": layer.name,
                "layer_type": type(layer).__name__,
                "trainable": layer.trainable
            }

            if hasattr(layer, 'units'):
                layer_info["units"] = layer.units
                total_units += layer.units

            if hasattr(layer, 'kernel_size'):
                layer_info["kernel_size"] = layer.kernel_size

            if hasattr(layer, 'activation'):
                if layer.activation == tf.keras.activations.relu:
                    layer_info["activation"] = "ReLU"
                elif layer.activation == tf.keras.activations.sigmoid:
                    layer_info["activation"] = "Sigmoid"
                elif layer.activation == tf.keras.activations.tanh:
                    layer_info["activation"] = "Tanh"
                elif layer.activation == tf.keras.activations.softmax:
                    layer_info["activation"] = "Softmax"
                else:
                    layer_info["activation"] = layer.activation.__name__

            layers_info.append(layer_info)

        input_shape = model.input_shape if hasattr(
            model, 'input_shape') else None
        output_shape = model.output_shape if hasattr(
            model, 'output_shape') else None

        param_info = [
            {"name": weight.name, "shape": weight.shape,
                "trainable": weight.trainable}
            for weight in model.weights
        ]

        model_info = {
            "num_params": num_params,
            "total_units": total_units,
            "layers_info": layers_info,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "params": param_info,
        }
        return model_info

    def calculate_gradients(self,
                            sample: TensorType,
                            noise: List[TensorVariableType],
                            construct_noise_fn: Callable,
                            target_vector: TensorType,
                            loss: Loss,
                            preprocess_fn: Optional[Callable] = None
                            ) -> Tuple[TensorType, TensorType]:
        has_batch_dim = self.tensor_ops.has_batch_dim(target_vector)
        num_classes = target_vector.shape[1] if has_batch_dim else target_vector.shape[0]

        with tf.GradientTape(persistent=True) as tape:
            for n in noise:
                tape.watch(n)

            perturbation = construct_noise_fn(noise)

            if preprocess_fn is not None:
                input = preprocess_fn(sample + perturbation)
            else:
                input = sample + perturbation

            logits = self.model(input)
            preds = self.act(logits)
            loss_value = loss.calculate(
                target=target_vector, predictions=preds, logits=logits, noise=perturbation) if not hasattr(loss, "__dummy__") else None

            if self.efficient_mode is not None:
                indexes = self._get_efficient_mode_indexes(preds, target_vector, num_classes, has_batch_dim)
                individual_logits = [
                    (logits[slice(0, 1), slice(i, i + 1)] if has_batch_dim else logits[slice(i, i + 1)])
                    if i in indexes else None
                    for i in range(num_classes)
                ]
                # individual_logits = []
                # for i in range(num_classes):
                #     if i in indexes:
                #         if has_batch_dim:
                #             val = logits[0, i]
                #         else:
                #             val = logits[i]
                #         individual_logits.append(val)
                #     else:
                #         individual_logits.append(None)

        if self._compute_jacobian:
            if self.efficient_mode is not None:
                logit_grads = self._get_efficient_mode_grads(tape, individual_logits, noise, has_batch_dim)
            else:
                logit_grads = tape.jacobian(logits, noise)
        else:
            logit_grads = None

        grad_wrt_loss = tape.gradient(
            loss_value, noise) if loss_value is not None else None

        if not hasattr(loss, "__dummy__") and grad_wrt_loss is None:
            raise IndifferentiabilityError()

        return grad_wrt_loss, logit_grads, logits, preds

    def predict(self,
                x: TensorType | np.ndarray,
                preprocess_fn: Optional[Callable] = None
                ) -> TensorType:
        if preprocess_fn is not None:
            sample = preprocess_fn(x)
        else:
            sample = x
        return self.act(self.model(sample, training=False))

    def forward(self,
                x: TensorType | np.ndarray
                ) -> TensorType:
        return self.act(self.model(x))
    
    def _get_efficient_mode_indexes(self, 
                                    preds, 
                                    target_vector, 
                                    num_classes, 
                                    has_batch_dim
                                    ) -> List[int]:
        if len(self.efficient_mode_indexes) != 0 and max(self.efficient_mode_indexes) >= num_classes:
            raise ValueError(f"Efficient mode indexes {self.efficient_mode_indexes} exceed number of classes {num_classes}. "
                                "The indexes must lie within the range of the number of classes.")
        indexes = set(self.efficient_mode_indexes) if self.efficient_mode_indexes else set()
        indexes.add(int(tf.argmax(target_vector[0] if has_batch_dim else target_vector).numpy()))
        indexes.update([int(i) for i in tf.argsort(preds[0] if has_batch_dim else preds, direction='DESCENDING')[:self.efficient_mode].numpy().tolist()])

        return indexes
    
    def _get_efficient_mode_grads(self, 
                                  tape, 
                                  individual_logits, 
                                  noise, 
                                  has_batch_dim
                                  ) -> List[TensorType]:
        logit_grads = [tape.gradient(logit, noise)[0] if logit is not None else None for logit in individual_logits]

        # sample_grad = tape.gradient([i for i in individual_logits if i is not None][0], noise)[0]
        # logit_grads = [tape.gradient(logit, noise)[0] if logit is not None else tf.zeros_like(sample_grad) for logit in individual_logits]
        # logit_grads = tf.stack(logit_grads, axis=0)
        # if has_batch_dim:
        #     logit_grads = self.tensor_ops.add_batch_dim(logit_grads, axis=0)

        return logit_grads
    

class ALModelNumpy(ALModelBase):
    def __init__(self,
                 model: Callable,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 *args,
                 **kwargs) -> None:       
        super().__init__(model, 
                         framework="numpy", 
                         efficient_mode=None, 
                         efficient_mode_indexes=None, 
                         gradient_estimator=gradient_estimator)
        self.max_queries = 1000
        self.query_count = 0

    def get_info(self, model: Callable) -> Dict[str, Any]:
        return {"model_type": "function", "framework": "numpy"}

    def calculate_gradients(self,
                            sample: np.ndarray,
                            noise: List[np.ndarray],
                            construct_perturbation_fn: Callable,
                            target_vector: np.ndarray,
                            loss: Loss,
                            ) -> Tuple[np.ndarray, Optional[np.ndarray]]: 
        grad_wrt_loss = self.gradient_estimator.calculate(
            sample=sample,
            noise=deepcopy(noise),
            target_vector=target_vector,
            predict_fn=self.predict,
            construct_perturbation_fn=construct_perturbation_fn,
            loss=loss,
        )

        logits = None
        logit_grads = None 
        preds = self.model([sample + construct_perturbation_fn(noise)])

        return grad_wrt_loss, logit_grads, logits, preds

    def predict(self,
                x: np.ndarray,
                *args,
                **kwargs,
                ) -> Union[np.ndarray, Tuple[int, float]]:
        self.query_count += 1
        return self.model(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.query_count += 1
        return self.model(x)
    
    def set_max_queries(self, max_queries: int) -> None:
        self.max_queries = max_queries

    def reset_query_count(self) -> None:
        self.query_count = 0



class ALModel(ALModelBase, metaclass=ALModelMeta):
    def __init__(self,
                 model: ModelType,
                 framework: str,
                 efficient_mode: Optional[int] = None,
                 efficient_mode_vector: Optional[np.ndarray] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model, 
                         framework, 
                         efficient_mode=efficient_mode, 
                         efficient_mode_indexes=efficient_mode_vector,
                         gradient_estimator=gradient_estimator)
