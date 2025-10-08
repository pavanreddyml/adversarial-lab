from abc import ABC

import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from adversarial_lab.callbacks import *
from adversarial_lab.core import ALModel
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.analytics import AdversarialAnalytics, Tracker
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.gradient_estimator import GradientEstimator, DummyGradientEstimator
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing, PreprocessingFromFunction


from adversarial_lab.core.noise_generators import NoiseGenerator, TensorNoiseGenerator, TextNoiseGenerator
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator

from adversarial_lab.core.types import TensorType
from typing import Union, List, Optional, Literal, Callable


NoiseGeneratorType = Union[TensorNoiseGenerator, TextNoiseGenerator]

class AdversarialAttackerBase(ABC):
    """
    Base class for white-box adversarial attack. Subclasses must implement specific attack methods.
    """

    @property
    def _compatible_noise_generators(self) -> List[NoiseGenerator]:
        """
        Returns a list of compatible noise generator names for this attack.
        Subclasses should override this property to specify compatible noise generators.
        """
        return ()

    @property
    def _compatible_trackers(self) -> List[Tracker]:
        """
        Returns a list of compatible optimizer names for this attack.
        Subclasses should override this property to specify compatible optimizers.
        """
        return ()

    def __init__(self,
                 model: ALModel,
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 constraints: Optional[List[PostOptimizationConstraint]] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 efficient_mode: Optional[int] = False,
                 efficient_mode_indexes: Optional[List[int]] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 verbose: int = 1,
                 *args,
                 **kwargs
                 ) -> None:
        """
        Initializes the WhiteBoxAttack with the given model, loss function, optimizer, noise generator,
        preprocessing, constraints, and analytics.

        Args:
            model (ALModel, torch.nn.Module, tf.keras.Model, tf.keras.Sequential): 
                The model to be attacked.
            loss (Union[str, Loss], optional): 
                The loss function to use for the attack. If a string, retrieves the loss from the LossRegistry.
            optimizer (Union[str, Optimizer]): 
                The optimizer to use for the attack. If a string, retrieves the optimizer from the OptimizerRegistry.
            noise_generator (NoiseGenerator, optional): 
                The noise generator to use for the attack. If None, `AdditiveNoiseGenerator` will be used.
            preprocessing (Preprocessing, optional): 
                The preprocessing method to use. If None, `NoPreprocessing` will be used.
            constraints (List[PostOptimizationConstraint], optional): 
                A list of constraints to apply on the noise after the attack, in order. If None, no constraints will be applied.
            analytics (AdversarialAnalytics, optional): 
                The analytics module for tracking attack metrics. If None, no metrics will be tracked.
            *args (tuple): 
                Additional positional arguments.
            **kwargs (dict): 
                Additional keyword arguments.

        Raises:
            TypeError: If:
                - `loss` is not an instance of `Loss` or a valid string.
                - `optimizer` is not an instance of `Optimizer` or a valid string.
                - `noise_generator` is not an instance of `NoiseGenerator`.
                - `preprocessing` is not an instance of `Preprocessing`.
                - `constraints` are not instances of `PostOptimizationConstraint`.
                - `analytics` is not an instance of `AdversarialAnalytics`.

        Notes:
                - `compute_jacobian` determines whether the Jacobian is computed. 
                This is set by `noise_generator` and defaults to `False`. 
                To enable, set `requires_jacobian=True` whilie initializing noise generator.
        """
        if isinstance(model, ALModel):
            self.model = model
            self.model.efficient_mode = efficient_mode
            self.model.efficient_mode_indexes = efficient_mode_indexes or []
            self.model.gradient_estimator = gradient_estimator
        else:
            self.model = ALModel(model,
                                 efficient_mode=efficient_mode,
                                 efficient_mode_indexes=efficient_mode_indexes,
                                 gradient_estimator=gradient_estimator)
        self.framework: str = self.model.framework

        self._optimizer_arg = optimizer
        self._initialize_preprocessing(preprocessing)
        self._initialize_optimizer(self._optimizer_arg)
        self._initialize_loss(loss)
        self._initialize_noise_generator(noise_generator)
        self._initialize_constraints(constraints)
        self._initialize_callbacks(callbacks)
        self._initialize_analytics(analytics)
        self._initialize_gradient_estimator(gradient_estimator)

        self.tensor_ops = TensorOps(framework=self.framework)

        self.verbose = verbose

        self.progress_bar: Optional[tqdm] = None

    def attack(self,
               epochs: int,
               *args,
               **kwargs
               ) -> np.ndarray:
        """
        Launch the attack process.

        Parameters:
            epochs (int): The number of epochs to run the attack.
            *args (tuple): Additional positional arguments.
            **kwargs (dict): Additional keyword arguments, including:
                - verbose (int): Verbosity level. 
                    - `0`: No output.
                    - `1`: Progress bar.
                    - `2+`: Detailed text output.

        Returns:
            np.ndarray: The adversarial noise generated during the attack.
        """
        self.model.reset_query_count()
        self._initialize_optimizer(self._optimizer_arg)
        self._reset_callbacks()

        if "ipykernel" in sys.modules or "google.colab" in sys.modules:
            from IPython.display import clear_output
            clear_output(wait=True)

        if hasattr(self, "progress_bar"):
            del self.progress_bar

        self.progress_bar = tqdm(
            total=epochs, desc="Attacking", leave=True, disable=(self.verbose == 0))
        
    def _initialize_preprocessing(self, preprocessing):
        """
        Initializes the preprocessing pipeline for the attack. If no preprocessing is provided,
        `NoPreprocessing` is used by default.

        Parameters:
        ----------
        preprocessing : Preprocessing, optional
            The preprocessing pipeline to apply before generating adversarial noise.

        Raises:
        -------
        TypeError
            If the preprocessing argument is not an instance of `Preprocessing`.
        """
        if preprocessing is None:
            self.preprocessing = NoPreprocessing()
        elif callable(preprocessing):
            self.preprocessing = PreprocessingFromFunction.create(
                preprocessing_function=preprocessing)
        elif isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        else:
            raise TypeError(
                f"Invalid type for preprocessing: '{type(preprocessing)}'")

    def _initialize_optimizer(self, optimizer):
        """
        Initialize the optimizer for the attack.

        The optimizer can be specified as either a string or an `Optimizer` instance.

        Parameters:
            optimizer (Union[str, Optimizer]): The optimizer to use for updating adversarial noise.

        Raises:
            TypeError: If `optimizer` is neither a string nor an instance of `Optimizer`.

        Notes:
            - If `optimizer` is a string, it will be retrieved from the `OptimizerRegistry`.
            - If `optimizer` is an instance of `Optimizer`, it will be used directly.
            - The framework of the optimizer is set to match the model's framework.
            - The optimizer is deep-copied to avoid modifying the original instance. This will be used
            to reinitialize the optimizer for each attack to track the new Noise Tensor
        """

        optimizer_copy = deepcopy(optimizer)

        if isinstance(optimizer_copy, str):
            optimizer_class = OptimizerRegistry.get(optimizer_copy)
            self.optimizer = optimizer_class()
        elif isinstance(optimizer_copy, Optimizer):
            self.optimizer = optimizer_copy
        else:
            raise TypeError(
                f"Invalid type for optimizer: '{type(optimizer_copy)}'")

        self.optimizer.set_framework(self.framework)

    def _initialize_loss(self, loss):
        """
        Initialize the loss function for the attack.

        The loss function can be specified as either a string or a `Loss` instance. The framework
        for the loss function is set to match the model's framework.

        Parameters:
            loss (Union[str, Loss]): The loss function to use for the attack.

        Raises:
            TypeError: If `loss` is neither a string nor an instance of `Loss`.

        Notes:
            - If `loss` is a string, it will be retrieved from the `LossRegistry`.
            - If `loss` is an instance of `Loss`, it will be used directly.
            - The framework of the loss function is set to match the model's framework.
        """

        if isinstance(loss, (str, type(None))):
            loss_class = LossRegistry.get(loss)
            self.loss = loss_class()
        elif isinstance(loss, Loss):
            self.loss = loss
        else:
            raise TypeError(f"Invalid type for loss: '{type(loss)}'")

        self.loss.set_framework(self.framework)

    def _initialize_noise_generator(self, noise_generator):
        """
        Initialize the noise generator for the attack. If no noise generator is provided, 
        `AdditiveNoiseGenerator` is used by default.

        Parameters:
            noise_generator (NoiseGenerator, optional): The noise generator to use for the attack.

        Raises:
            TypeError: If `noise_generator` is not an instance of `NoiseGenerator`.
            ValueError: If `noise_generator` is not a valid type.

        Notes:
            - If `noise_generator` is None, an instance of `AdditiveNoiseGenerator` with a uniform 
            distribution and scale (-0.05, 0.05) is used by default.
            - If `noise_generator` is provided, it must be an instance of `NoiseGenerator`.
            - The framework of the noise generator is set to match the model's framework.
            - The model's `compute_jacobian` flag is set based on whether the noise generator 
            requires the Jacobian.
        """

        if noise_generator is None:
            if TensorNoiseGenerator in self._compatible_noise_generators:
                self.noise_generator: NoiseGeneratorType = AdditiveNoiseGenerator(
                    scale=(-0.05, 0.05), dist='uniform')
            elif TextNoiseGenerator in self._compatible_noise_generators:
                raise NotImplementedError(
                    "Default noise generator for text is not implemented. Please provide a valid noise generator.")
            else:
                raise ValueError(
                    "No compatible noise generator found. Please provide a valid noise generator.")
        elif isinstance(noise_generator, NoiseGenerator):
            if not isinstance(noise_generator, self._compatible_noise_generators):
                raise ValueError(
                    f"Invalid noise generator type: '{type(noise_generator)}'. Must be one of {self._compatible_noise_generators}.")
            self.noise_generator: NoiseGeneratorType = noise_generator
        else:
            raise TypeError(
                f"Invalid type for noise_generator: '{type(noise_generator)}'")
        
        self.noise_generator.set_framework(self.framework)
        self.model.set_compute_jacobian(self.noise_generator.requires_jacobian)

    def _initialize_constraints(self,
                                constraints: List[PostOptimizationConstraint]):
        """
        Initialize the constraints for the attack. If no constraints are provided, an empty list is used by default.

        Parameters:
            constraints (List[PostOptimizationConstraint], optional): 
                The constraints to apply to the adversarial noise after optimization.

        Raises:
            TypeError: If any element in `constraints` is not an instance of `PostOptimizationConstraint`.

        Notes:
            - If `constraints` is None, an empty list is used as the default.
            - Each constraint must be an instance of `PostOptimizationConstraint`.
            - The framework of each constraint is set to match the model's framework.
        """
        if constraints is None:
            constraints = []

        for contraint in constraints:
            if not isinstance(contraint, PostOptimizationConstraint):
                raise TypeError(
                    f"Invalid type for constraints: '{type(contraint)}'")

        for constraint in constraints:
            constraint.set_framework(self.framework)

        self.constraints = constraints

    def _initialize_analytics(self, analytics):
        """
        Initialize the analytics module for tracking attack metrics. If no analytics module is provided, 
        a default `AdversarialAnalytics` instance is created.

        Parameters:
            analytics (AdversarialAnalytics, optional): 
                The analytics module to use for tracking attack metrics.

        Raises:
            TypeError: If `analytics` is not an instance of `AdversarialAnalytics`.

        Notes:
            - If `analytics` is None, a default `AdversarialAnalytics` instance is initialized 
            with no database, no trackers, and no table name. Will not track any metrics.
        """
        if analytics is not None:
            if not isinstance(analytics, AdversarialAnalytics):
                raise TypeError(
                    "analytics must be an instance of AdversarialAnalytics")
            self.analytics = analytics
        else:
            self.analytics = AdversarialAnalytics(
                db=None, trackers=[], table_name=None)

        self.analytics_function_map = {
            "pre_train": self.analytics.update_pre_attack_values,
            "post_batch": self.analytics.update_post_batch_values,
            "post_epoch": self.analytics.update_post_epoch_values,
            "post_train": self.analytics.update_post_attack_values
        }

    def _initialize_callbacks(self, callbacks: List[Callback]):
        """
        Initialize the callbacks for the attack. If no callbacks are provided, an empty list is used by default.

        """

        if callbacks is None:
            callbacks = []

        if not isinstance(callbacks, list):
            raise TypeError("callbacks must be a list of Callback instances.")

        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(
                    f"Invalid type for callback: '{type(callback)}'. Must be an instance of Callback.")

            if isinstance(callback, ChangeParams):
                for key in callback.params.get("optimizer", {}).keys():
                    if not self.optimizer.has_param(key):
                        raise ValueError(
                            f"Optimizer does not have parameter '{key}' to change. Update ChangeParams to use a valid parameter.")

                for key in callback.params.get("loss", {}).keys():
                    if not self.loss.has_param(key):
                        raise ValueError(
                            f"Loss does not have parameter '{key}' to change. Update ChangeParams to use a valid parameter.")

                for i, penalty in enumerate(callback.params.get("penalties", [])):
                    for key in penalty.keys():
                        if not self.loss.penalties[i].has_param(key):
                            raise ValueError(
                                f"Penalty {repr(penalty)} does not have parameter '{key}' to change. Update ChangeParams to use a valid parameter.")

        self.callbacks = callbacks

    def _initialize_gradient_estimator(self, gradient_estimator: Optional[GradientEstimator]):
        if gradient_estimator is None:
            self.gradient_estimator = DummyGradientEstimator()
            return
        elif not isinstance(gradient_estimator, GradientEstimator):
            raise TypeError(
                f"Invalid type for gradient_estimator: '{type(gradient_estimator)}'. Must be an instance of GradientEstimator.")
        else:
            self.gradient_estimator = gradient_estimator

        self.gradient_estimator.set_framework(self.framework)

    
    def _reset_callbacks(self):
        """
        Reset the state of all callbacks.
        """
        for callback in self.callbacks:
            callback.reinitialize()

    def _apply_constrains(self, noise, sample):
        """
        Apply all constraints to the noise tensor.

        Parameters:
            noise (tf.Variable, torch.Tensor): The noise tensor to which constraints are applied.

        Notes:
            - Each constraint in `self.constraints` is applied in sequence.
        """
        for constraint in self.constraints:
            for n in noise:
                constraint.apply(noise=n, sample=sample)

    def _update_analytics(self,
                          when: Literal["pre_train", "post_batch", "post_epoch", "post_train"],
                          loss=None,
                          original_sample=None,
                          preprocessed_sample=None,
                          noise_preprocessed_sample=None,
                          predictions=None,
                          write_only=False,
                          *args,
                          **kwargs):
        if when not in ["pre_train", "post_batch", "post_epoch", "post_train"]:
            raise ValueError(
                "Invalid value for 'when'. Must be one of ['pre_train', 'post_batch', 'post_epoch', 'post_train'].")

        if when in ["post_epoch", "post_batch"] or write_only:
            epoch = kwargs.get("epoch", None)
            if epoch is None:
                raise ValueError(
                    "Epoch number must be provided for 'post_epoch' and 'post_batch' analytics.")
        elif when == "pre_train":
            epoch = 0
        elif when == "post_train":
            epoch = 99999999

        if write_only:
            self.analytics.write(epoch_num=epoch)
            return

        analytics_func = self.analytics_function_map[when]

        analytics_func(
            loss=loss,
            original_sample=original_sample,
            preprocessed_sample=preprocessed_sample,
            noise_preprocessed_sample=noise_preprocessed_sample,
            predictions=predictions,
            *args,
            **kwargs
        )

        if when != "post_batch":
            self.analytics.write(epoch_num=epoch)

    def _update_progress_bar(self,
                             predictions: np.ndarray,
                             true_class: int,
                             target_class: int):
        predicted_class = np.argmax(predictions)
        predicted_class_confidence = predictions[predicted_class]

        true_class_confidence = predictions[true_class]
        target_class_confidence = predictions[target_class]

        loss = self.loss.get_total_loss()

        self.progress_bar.update(1)
        if self.verbose == 2:
            self.progress_bar.set_postfix(
                {
                    'Loss': loss,
                    'Prediction (score)': f"{predicted_class}({predicted_class_confidence:.3f})",
                })
        if self.verbose >= 3:
            self.progress_bar.set_postfix(
                {
                    'Loss': loss,
                    'Prediction (score)': f"{predicted_class}({predicted_class_confidence:.3f})",
                    'Original Class (score)': f"{true_class}({true_class_confidence:.3f})",
                    'Target Class (score)': f"{target_class}({target_class_confidence:.3f})",
                })

    def _apply_callbacks(self,
                         predictions,
                         true_class,
                         target_class,
                         when: Literal["post_epoch"] = "post_epoch"):
        callback_list = {}

        for callback in self.callbacks:
            if not callback.enabled:
                continue

            call_back_results = callback.on_epoch_end(
                predictions=predictions,
                true_class=true_class,
                target_class=target_class,
                loss=self.loss,
            )

            if call_back_results is not None:
                if isinstance(callback, ChangeParams):
                    callback.apply_changes(
                        optimizer=self.optimizer, loss=self.loss)
                else:
                    callback_list.update(call_back_results)

                if callback.blocking:
                    break

        return callback_list

    def _is_hard_vector(self, vector: TensorType) -> bool:
        return np.all(np.logical_or(vector.numpy() == 0, vector.numpy() == 1))
