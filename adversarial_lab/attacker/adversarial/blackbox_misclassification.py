import random
import warnings
import numpy as np

from .base_adversarial_attacker import AdversarialAttackerBase
from adversarial_lab.core.losses import Loss
from adversarial_lab.callbacks import Callback
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.exceptions import VectorDimensionsError
from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.analytics import AdversarialAnalytics, Tracker
from adversarial_lab.core.gradient_estimator import GradientEstimator
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.core.noise_generators import NoiseGenerator, TensorNoiseGenerator

from typing import Optional, List, Callable, Literal, Union


class BlackBoxMisclassificationAttack(AdversarialAttackerBase):
    """
    BlackBoxMisclassificationAttack generates adversarial examples that force misclassification.
    """

    @property
    def _compatible_noise_generators(self) -> List[NoiseGenerator]:
        return (TensorNoiseGenerator, )

    @property
    def _compatible_trackers(self) -> List[Tracker]:
        return ()

    def __init__(self,
                 model: Callable,
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 constraints: Optional[PostOptimizationConstraint] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: int = 1,
                 max_queries: int = 10000,
                 *args,
                 **kwargs) -> None:
        super().__init__(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         noise_generator=noise_generator,
                         gradient_estimator=gradient_estimator,
                         constraints=constraints,
                         analytics=analytics,
                         callbacks=callbacks,
                         verbose=verbose,
                         max_queries=max_queries,
                         *args,
                         **kwargs)

    def attack(self,
               sample: np.ndarray,
               target_class: int = None,
               target_vector: np.ndarray = None,
               strategy: Literal['spread', 'uniform', 'random'] = "random",
               binary_threshold: float = 0.5,
               epochs=10,
               *args,
               **kwargs
               ) -> np.ndarray:
        super().attack(epochs, *args, **kwargs)

        sample_to_attack, predictions, noise_meta, target_vector, true_class, target_class, query_type = self._initialize_attack(
            sample=sample,
            target_class=target_class,
            target_vector=target_vector,
            strategy=strategy,
            binary_threshold=binary_threshold,
        )

        if query_type == "class" and self.model.gradient_estimator is not None:
            NotImplementedError(
                "BlackBoxMisclassificationAttack does not support gradient estimation for queries that return class labels and confidence. "
                "Please use a query that returns prediction confidences or pass 'gradient_estimator=None' to the attack.")

        # Initial stats
        self._update_analytics(
            when="pre_train",
            loss=self.loss,
            original_sample=sample,
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=self.tensor_ops.remove_batch_dim(predictions),
            on_original=True,
        )

        for epoch in range(epochs):
            grads, jacobian, logits, preds = self.model.calculate_gradients(
                sample=sample_to_attack,
                noise=noise_meta,
                construct_perturbation_fn=self.noise_generator.construct_noise,
                target_vector=target_vector,
                loss=self.loss
                )

            self.noise_generator.update(
                noise_meta=noise_meta,
                optimizer=self.optimizer,
                grads=grads,
                jacobian=jacobian,
                logits=logits,
                predictions=preds,
                target_vector=target_vector,
                true_class=true_class,
                target_class=target_class,
            )

            self._apply_constrains(noise_meta, sample_to_attack)

            # Stats
            predictions = self.model.predict(x=[sample_to_attack+self.noise_generator.construct_noise(noise_meta)])[0]

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
                noise=self.noise_generator.get_noise(noise_meta),
                predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(
                    predictions)),
                on_original=True,
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
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=self.tensor_ops.numpy(self.tensor_ops.remove_batch_dim(predictions)),
            on_original=True,
        )

        return self.noise_generator.get_noise(noise_meta), noise_meta

    def _initialize_attack(self,
                           sample: np.ndarray,
                           target_class: int = None,
                           target_vector: np.ndarray = None,
                           strategy: Literal['spread',
                                             'uniform', 'random'] = "random",
                           binary_threshold: float = 0.5
                           ) -> None:
        query_type = None
        sample_to_attack = self.tensor_ops.tensor(sample)

        if self.tensor_ops.has_batch_dim(sample_to_attack):
            warnings.warn("For blackbox attacks, the input sample should not have a batch dimension. " 
                          "If your using batch dimension, it may lead to unexpected results. "
                          "If the first dimension is of size 1 and not a batch dimension, ignore this warning.")

        noise_meta = self.noise_generator.generate_noise_meta(sample_to_attack)

        predictions = self.model.predict(
            x=[sample_to_attack+self.noise_generator.construct_noise(noise_meta)])[0]  # Testing if noise can be applied to the preprocessed image
        predictions = self.model.predict(x=[sample_to_attack])

        if not isinstance(predictions, list):
            raise TypeError(
                "`predict_fn` must return List[np.ndarray] or a List[Tuple[Tuple[prediction_class, prediction_confidence]]].")
        predictions = predictions[0]

        if not isinstance(predictions, np.ndarray) and not isinstance(predictions, tuple):
            raise TypeError(
                "The `predict_fn` return value must be a list containing either numpy arrays (List[np.ndarray]) "
                "or tuples of prediction class and confidence (List[Tuple[Tuple[int, float]]])."
                "The element in the returned list does not match either of these types.")

        if isinstance(predictions, np.ndarray):
            query_type = "vector"
            if self.tensor_ops.has_batch_dim(predictions):
                raise VectorDimensionsError(
                    "`predict_fn` must return a 1d array of prediction confidences without batch dimension.")
            if target_vector is not None:
                target_vector = self.tensor_ops.tensor(target_vector)
            elif target_class:
                if target_class >= len(predictions):
                    raise ValueError(
                        "target_class exceeds the dimension of the outputs.")
                target_vector = np.zeros(
                    shape=(len(predictions), ))
                target_vector[target_class] = 1
            else:
                target_vector = self._get_target_vector(
                    predictions=predictions,
                    strategy=strategy
                )
            target_vector = target_vector
            true_class = np.argmax(predictions)
            target_class = np.argmax(target_vector)
            

        if isinstance(predictions, tuple):
            query_type = "class"
            if len(predictions) != 1:
                raise NotImplementedError(
                    "Blackbox Attacks currently do not support multi-output models. "
                    "Please provide a single output from the model.")
            for pred in predictions:
                if len(pred) != 2 or not isinstance(pred, tuple) or not isinstance(pred[0], (int, np.integer)) or not isinstance(pred[1], (float, np.floating)):
                    raise ValueError(
                        "`predict_fn` must return a Tuple[Tuple[prediction_class, prediction_confidence]].")
            if target_vector is not None:
                raise ValueError(
                    "target_vector cannot be used with query_type 'class'. Use target_class instead.")
            target_vector = None   
            true_class = predictions[0][0]
            target_class = target_class

        return sample_to_attack, predictions, noise_meta, target_vector, true_class, target_class, query_type

    def _get_target_vector(self,
                           predictions: np.ndarray,
                           strategy: Literal['spread', 'uniform', 'random']
                           ) -> np.ndarray:
        num_classes = predictions.shape[0]
        true_class = np.argmax(predictions)

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

        if predictions.shape != target_vector.shape:
            raise VectorDimensionsError(
                "target_vector must be the same size as the outputs.")

        return target_vector

