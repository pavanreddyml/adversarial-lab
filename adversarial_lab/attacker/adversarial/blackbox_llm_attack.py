from typing import Callable, Optional, List, Union

from .base_adversarial_attacker import AdversarialAttackerBase
from adversarial_lab.core.noise_generators import NoiseGenerator, TextNoiseGenerator
from adversarial_lab.core.losses import Loss
from adversarial_lab.core.optimizers import Optimizer
from adversarial_lab.analytics import AdversarialAnalytics, Tracker
from adversarial_lab.core.gradient_estimator import GradientEstimator
from adversarial_lab.core.constraints import PostOptimizationConstraint
from adversarial_lab.callbacks import Callback


class BlackBoxLLMAttack(AdversarialAttackerBase):
    """Attack that tries to inject adversarial text into LLM prompts."""

    @property
    def _compatible_noise_generators(self) -> List[NoiseGenerator]:
        return (TextNoiseGenerator,)

    @property
    def _compatible_trackers(self) -> List[Tracker]:
        return ()
    
    @property
    def _compatible_gradient_estimators(self) -> List[GradientEstimator]:
        return ()

    def __init__(self,
                 model: Callable[[str], str],
                 optimizer: Union[str, Optimizer],
                 loss: Optional[Union[str, Loss]] = None,
                 noise_generator: Optional[NoiseGenerator] = None,
                 gradient_estimator: Optional[GradientEstimator] = None,
                 constraints: Optional[PostOptimizationConstraint] = None,
                 analytics: Optional[AdversarialAnalytics] = None,
                 callbacks: Optional[List[Callback]] = None,
                 verbose: int = 1,
                 max_queries: int = 1000,
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

    def _update_progress_bar_text(self, prediction: str) -> None:
        self.progress_bar.update(1)
        if self.verbose >= 2:
            shortened = prediction.replace("\n", " ")[:30]
            self.progress_bar.set_postfix({"Prediction": shortened})

    def attack(self,
               prompt: str,
               target: Optional[str] = None,
               epochs: int = 10,
               *args,
               **kwargs):
        """Run the prompt injection attack."""
        super().attack(epochs, *args, **kwargs)

        noise_meta = self.noise_generator.generate_noise_meta(prompt)
        best_noise = self.noise_generator.get_noise(noise_meta)
        best_score = float("inf")
        prediction = self.model.predict(prompt)

        self._update_analytics(
            when="pre_train",
            loss=self.loss,
            original_sample=prompt,
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=prediction,
        )

        for epoch in range(epochs):
            adv_prompt = self.noise_generator.apply_noise(
                prompt, self.noise_generator.construct_noise(noise_meta))
            prediction = self.model.predict(adv_prompt)

            if not getattr(self.loss, "__dummy__", False):
                loss_val = self.loss.calculate(
                    target=target,
                    predictions=prediction,
                    logits=None,
                    from_logits=False,
                )
            else:
                loss_val = None

            self.noise_generator.update(
                noise_meta=noise_meta,
                optimizer=self.optimizer,
                grads=None,
                jacobian=None,
                logits=None,
                predictions=prediction,
                target_vector=target,
                true_class=None,
                target_class=None,
            )

            self._apply_constrains(noise_meta, prompt)

            if loss_val is not None and loss_val < best_score:
                best_score = loss_val
                best_noise = self.noise_generator.get_noise(noise_meta)

            self._update_progress_bar_text(prediction)
            self._update_analytics(
                when="post_epoch",
                epoch=epoch + 1,
                loss=self.loss,
                original_sample=prompt,
                noise=self.noise_generator.get_noise(noise_meta),
                predictions=prediction,
            )

            callbacks_data = self._apply_callbacks(
                predictions=prediction,
                true_class=None,
                target_class=None,
            )
            if "stop_attack" in callbacks_data:
                break

        self._update_analytics(
            when="post_train",
            loss=self.loss,
            original_sample=prompt,
            noise=self.noise_generator.get_noise(noise_meta),
            predictions=prediction,
        )

        return best_noise, noise_meta