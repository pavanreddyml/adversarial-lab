import random
import numpy as np

import warnings

from .base_adversarial_attacker import AdversarialAttackerBase
from adversarial_lab.core import ALModel
from adversarial_lab.core.losses import Loss
from adversarial_lab.callbacks import Callback
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.exceptions import VectorDimensionsError
from adversarial_lab.core.preprocessing import Preprocessing
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.analytics import AdversarialAnalytics, Tracker
from adversarial_lab.core.constraints import PostOptimizationConstraint

from adversarial_lab.core.noise_generators import NoiseGenerator, TensorNoiseGenerator

from typing import Optional, Union, Literal, List
from adversarial_lab.core.types import TensorType, ModelType


class WhiteBoxMisclassificationAttack(AdversarialAttackerBase):
    """
    WhiteBoxMisclassification is a white-box attack that generates adversarial examples by applying perturbations to the input sample.
    The attack aims to misclassify the input sample into a target class or target vector.
    It supports various strategies for generating the target vector and can handle binary and multi-class classification models.
    The attack uses a noise generator to create perturbations and an optimizer to update the noise during the attack process.
    """

    @property
    def _compatible_noise_generators(self) -> List[NoiseGenerator]:
        return (TensorNoiseGenerator, )

    @property
    def _compatible_trackers(self) -> List[Tracker]:
        return ()

    def __init__(self,
                 model: Union[ALModel, ModelType],
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 preprocessing: Optional[Preprocessing] = None,
                 constraints: Optional[PostOptimizationConstraint] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: int = 1,
                 efficient_mode: Optional[int] = None,
                 efficient_mode_indexes: Optional[List[int]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(model=model,
                         loss=loss,
                         optimizer=optimizer,
                         noise_generator=noise_generator,
                         preprocessing=preprocessing,
                         constraints=constraints,
                         analytics=analytics,
                         callbacks=callbacks,
                         verbose=verbose,
                         efficient_mode=efficient_mode,
                         efficient_mode_indexes=efficient_mode_indexes,
                         *args,
                         **kwargs)

    def attack(self,
               sample: Union[np.ndarray, TensorType],
               target_class: int = None,
               target_vector: Union[np.ndarray, TensorType] = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               binary_threshold: float = 0.5,
               epochs=10,
               on_original: bool = False,
               *args,
               **kwargs
               ) -> np.ndarray:
        if self.model._compute_jacobian and self.model.efficient_mode is None:
            warnings.warn(
                "The model is set to compute jacobian, but efficient mode is not enabled. "
                "This may lead to high memory usage and slow performance. Consider enabling efficient mode. "
                "Efficient mode computes logit gradients only for the required indexes, rather than "
                "the entire jacobian matrix. This can significantly reduce memory usage and speed up the attack. "
                "Ignore this warning if you require the full jacobian matrix. "
            )

        super().attack(epochs, *args, **kwargs)

        preprocessed_sample = self.tensor_ops.remove_batch_dim(
            self.preprocessing.preprocess(sample))

        sample_to_attack, predictions, noise_meta, target_vector = self._initialize_attack(
            sample=sample,
            target_class=target_class,
            target_vector=target_vector,
            strategy=strategy,
            binary_threshold=binary_threshold,
            on_original=on_original
        )

        true_class = np.argmax(
            self.tensor_ops.remove_batch_dim(predictions).numpy())
        target_class = np.argmax(
            self.tensor_ops.remove_batch_dim(target_vector).numpy())

        # Initial stats
        self._update_analytics(
            when="pre_train",
            loss=self.loss,
            original_sample=sample,
            preprocessed_sample=self.tensor_ops.numpy(preprocessed_sample),
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(predictions)),
            on_original=on_original,
        )

        for epoch in range(epochs):
            grads, jacobian, logits, preds = self.model.calculate_gradients(
                sample=sample_to_attack,
                noise=noise_meta,
                construct_noise_fn=self.noise_generator.construct_noise,
                target_vector=target_vector,
                loss=self.loss,
                preprocess_fn=self.preprocessing.preprocess if on_original else None)

            self.noise_generator.update(
                noise_meta=noise_meta,
                optimizer=self.optimizer,
                grads=grads,
                jacobian=jacobian,
                logits=logits,
                predictions=preds,
                target_vector=target_vector,
                true_class=true_class
            )

            self._apply_constrains(noise_meta, sample_to_attack)

            # Stats
            predictions = self.model.predict(
                x=sample_to_attack +
                self.noise_generator.construct_noise(noise_meta),
                preprocess_fn=self.preprocessing.preprocess if on_original else None
            )

            self._update_progress_bar(
                predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(
                    predictions)),
                true_class=true_class,
                target_class=target_class,
            )

            self._update_analytics(
                when="post_epoch",
                epoch=epoch+1,
                loss=self.loss,
                original_sample=sample,
                preprocessed_sample=preprocessed_sample,
                noise=self.noise_generator.get_noise(noise_meta),
                predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(
                    predictions)),
                on_original=on_original,
            )

            callbacks_data = self._apply_callbacks(
                predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(
                    predictions)),
                true_class=true_class,
                target_class=target_class,
            )

            if "stop_attack" in callbacks_data:
                break

        self._update_analytics(
            when="post_train",
            loss=self.loss,
            original_sample=sample,
            preprocessed_sample=preprocessed_sample,
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(predictions)),
            on_original=on_original,
        )

        return self.noise_generator.get_noise(noise_meta), noise_meta

    def _initialize_attack(self,
                           sample: Union[np.ndarray, TensorType],
                           target_class: int = None,
                           target_vector: Union[np.ndarray, TensorType] = None,
                           strategy: Literal['spread',
                                             'uniform', 'random'] = "random",
                           binary_threshold: float = 0.5,
                           on_original=False) -> None:
        sample = self.tensor_ops.tensor(sample)
        preprocessed_sample = self.preprocessing.preprocess(sample)
        sample_to_attack = sample if on_original else preprocessed_sample

        if not self.tensor_ops.has_batch_dim(preprocessed_sample):
            warnings.warn("The sample after preprocessing does not have a batch dimension. "
                          "Adding a batch dimension must be handled in the preprocessing function."
                          "If sample does not require a batch dimension, ignore this warning.")

        noise_meta = self.noise_generator.generate_noise_meta(
            sample_to_attack)

        predictions = self.model.predict(
            x=sample_to_attack +
            self.noise_generator.construct_noise(noise_meta),
            preprocess_fn=self.preprocessing.preprocess if on_original else None
        )  # Testing if noise can be applied to the preprocessed image
        predictions = self.model.predict(
            x=sample_to_attack,
            preprocess_fn=self.preprocessing.preprocess if on_original else None
        )

        if self.tensor_ops.has_batch_dim(predictions):
            has_batch_dim = True
        else:
            has_batch_dim = False
            predictions = self.tensor_ops.add_batch_dim(predictions)

        if self.model.model_info["output_shape"][1] == 1:
            if target_class:
                warnings.warn(
                    "target_class are automatically set for binary classification. Ignoring provided target_class.")
            true_class = (predictions >= binary_threshold).astype(int)[0]
            target_class = 1 - true_class if not target_vector else None
        else:
            true_class = np.argmax(predictions, axis=1)[0]

        if target_class is not None and target_vector is not None:
            raise ValueError(
                "target_class and target_vector cannot be used together.")

        if target_class:
            if target_class >= self.model.model_info["output_shape"][1]:
                raise ValueError(
                    "target_class exceeds the dimension of the outputs.")
            target_vector = np.zeros(
                shape=(self.model.model_info["output_shape"][1], ))
            target_vector[target_class] = 1

        if target_vector is not None:
            target_vector = self.tensor_ops.tensor(target_vector)
            if len(target_vector) != self.model.model_info["output_shape"][1]:
                raise VectorDimensionsError(
                    "target_vector must be the same size outputs.")
        else:
            if strategy not in ['spread', 'uniform', 'random']:
                raise ValueError(
                    "Invalid value for strategy. It must be 'spread', 'uniform', 'random'.")
            target_vector = self._get_target_vector(
                predictions, true_class, strategy)

        target_vector = self.tensor_ops.add_batch_dim(
            target_vector) if has_batch_dim else self.tensor_ops.remove_batch_dim(target_vector)

        return sample_to_attack, predictions, noise_meta, target_vector

    def _get_target_vector(self,
                           predictions: Union[np.ndarray, TensorType],
                           true_class: int,
                           strategy: Literal['spread', 'uniform', 'random']
                           ) -> Union[np.ndarray, TensorType]:
        num_classes = predictions.shape[1]

        if strategy == 'spread':
            target_vector = np.ones(num_classes) / (num_classes - 1)
            target_vector[true_class] = 1e-6
            target_vector /= target_vector.sum()
        elif strategy == 'uniform':
            target_vector = np.ones(num_classes) / num_classes
        elif strategy == 'random':
            random_class = random.choice(
                [i for i in range(num_classes) if i != true_class])
            target_vector = np.zeros(shape=(num_classes, ))
            target_vector[random_class] = 1

        target_vector = self.tensor_ops.tensor(target_vector)
        target_vector = self.tensor_ops.add_batch_dim(target_vector)

        if predictions.shape != target_vector.shape:
            raise VectorDimensionsError(
                "target_vector must be the same size as the outputs.")

        return target_vector
