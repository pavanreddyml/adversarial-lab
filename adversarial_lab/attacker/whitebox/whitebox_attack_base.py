from abc import ABC

import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy


from adversarial_lab.callbacks import *
from adversarial_lab.core import ALModel
from adversarial_lab.attacker import AttackerBase
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.losses import Loss, LossRegistry
from adversarial_lab.analytics import AdversarialAnalytics, Tracker
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.core.optimizers import Optimizer, OptimizerRegistry
from adversarial_lab.core.preprocessing import NoPreprocessing, Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator, TensorNoiseGenerator, TextNoiseGenerator
from adversarial_lab.core.noise_generators.tensor import AdditiveNoiseGenerator

from typing import Union, List, Optional, Literal
from adversarial_lab.core.types import TensorType


class WhiteBoxAttack(AttackerBase):
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
                 noise_generator: Optional[NoiseGenerator] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 constraints: Optional[List[PostOptimizationConstraint]] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 efficient_mode: Optional[int] = False,
                 efficient_mode_indexes: Optional[List[int]] = None,
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
        super().__init__(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         noise_generator=noise_generator,
                         constraints=constraints,
                         analytics=analytics,
                         callbacks=callbacks,
                         efficient_mode=efficient_mode,
                         verbose=verbose,
                         efficient_mode_indexes=efficient_mode_indexes,
                         *args,
                         **kwargs)
        self._initialize_preprocessing(preprocessing)

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
        elif isinstance(preprocessing, Preprocessing):
            self.preprocessing = preprocessing
        else:
            raise TypeError(
                f"Invalid type for preprocessing: '{type(preprocessing)}'")
