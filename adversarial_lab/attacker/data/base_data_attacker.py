from abc import ABC

import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from adversarial_lab.handlers import HandlerBase
from adversarial_lab.analytics import AdversarialAnalytics, Tracker

from adversarial_lab.core.noise_generators import NoiseGenerator, TensorNoiseGenerator, TextNoiseGenerator

from typing import Union, List, Optional, Literal


NoiseGeneratorType = Union[TensorNoiseGenerator, TextNoiseGenerator]

class DatasetAttackerBase(ABC):

    def __init__(self,
                 handler: HandlerBase,
                 noise_generator: NoiseGenerator,
                 analytics: Optional[AdversarialAnalytics] = None,
                 verbose: int = 1,
                 *args,
                 **kwargs
                 ) -> None:
        self._initialize_handler(handler)
        self._initialize_noise_generator(noise_generator)
        self._initialize_analytics(analytics)

        self.verbose = verbose

        self.progress_bar: Optional[tqdm] = None

    def attack(self,
               total_samples: int,
               *args,
               **kwargs
               ) -> np.ndarray:
        if hasattr(self, "progress_bar"):
            del self.progress_bar

        if "ipykernel" in sys.modules or "google.colab" in sys.modules:
            from IPython.display import clear_output
            clear_output(wait=True)

        self.progress_bar = tqdm(
            total=total_samples, desc="Poisoning", leave=True, disable=(self.verbose == 0))

    def _initialize_handler(self, handler: HandlerBase):
        self.handler = handler

    def _initialize_noise_generator(self, noise_generator: NoiseGenerator):
        self.noise_generator = noise_generator

    def _initialize_analytics(self, analytics: Optional[AdversarialAnalytics]):

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

    def _update_analytics(self,
                          when: Literal["pre_train", "post_batch", "post_epoch", "post_train"],
                          batch=None,
                          item_ids=None,
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
            batch=batch,
            item_ids=item_ids,
            *args,
            **kwargs
        )

        if when != "post_batch":
            self.analytics.write(epoch_num=epoch)

    def _update_progress_bar(self, batch, class_name, remaining, n=1, *args, **kwargs):
        self.progress_bar.update(n)
        if self.verbose == 2:
            self.progress_bar.set_postfix(
                {
                    "batch": batch,
                    "class": class_name,
                }
            )
        if self.verbose >= 3:
            self.progress_bar.set_postfix(
                {
                    "batch": batch,
                    "class": class_name,
                    "remaining": remaining
                    **kwargs
                }
            )

    def _update_progress_bar_desc(self, desc: str):
        if self.progress_bar is not None:
            self.progress_bar.set_description(desc)
