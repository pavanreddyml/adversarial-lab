
from typing import Dict, List, Optional
from .base_data_attacker import DatasetAttackerBase
import numpy as np

from adversarial_lab.handlers import HandlerBase
from adversarial_lab.analytics import AdversarialAnalytics, Tracker

from adversarial_lab.core.noise_generators import NoiseGenerator


class TrojanAttacker(DatasetAttackerBase):
    def __init__(self,
                 handler: HandlerBase,
                 noise_generator: NoiseGenerator,
                 copy_remaining: bool = False,
                 prefix: str = "trojaned_",
                 suffix: str = "",
                 analytics: Optional[AdversarialAnalytics] = None,
                 verbose: int = 1,
                 *args,
                 **kwargs):
        super().__init__(handler=handler,
                         noise_generator=noise_generator,
                         analytics=analytics,
                         verbose=verbose,
                         *args,
                         **kwargs)
        self.prefix = prefix
        self.suffix = suffix
        self.copy_remaining = copy_remaining

    def attack(self,
               class_to_poison: Optional[str] = None,
               amount_to_poison: Optional[float] = None,
               class_trojan_map: Optional[Dict[str, Dict[int, float]]] = None):
        super().attack(total_samples=self.handler.get_total_samples())

        class_trojan_map = self._validate_class_trojan_map(
            class_to_poison, amount_to_poison, class_trojan_map)
        batch_num = 0

        for class_to_poison, trojan_info in class_trojan_map.items():
            samples_to_poson = [int(trojan_info.get(tid, 0) * self.handler.get_num_entries_by_class(class_to_poison))
                                for tid in range(self.noise_generator.get_num_trojans())]
            while True:
                trojaned_samples = []
                batch = self.handler.get_batch(class_name=class_to_poison)

                if batch is None:
                    break

                if sum(samples_to_poson) == 0 and not self.copy_remaining:
                    break

                if sum(samples_to_poson) == 0 and self.copy_remaining:
                    trojaned_samples.extend(batch)
                    self._update_progress_bar_desc("Copying Remaining")
                    self._update_progress_bar(
                        batch=batch_num, class_name=class_to_poison, remaining=samples_to_poson, n=len(batch))
                else:
                    for batch_item in batch:
                        obj, idx, class_name = batch_item

                        trojan_idx = None
                        for i, count in enumerate(samples_to_poson):
                            if count != 0:
                                trojan_idx = i
                                break

                        if trojan_idx is None:
                            if not self.copy_remaining:
                                break
                            self._update_progress_bar_desc("Copying Remaining")
                            trojaned_samples.append((obj, idx, class_name))
                            break

                        trojaned_obj = self.noise_generator.apply_noise(
                            obj, trojan_id=trojan_idx)
                        trojaned_samples.append(
                            (trojaned_obj, f"{self.prefix}{idx}{self.suffix}", class_name))

                        samples_to_poson[trojan_idx] -= 1

                    self._update_progress_bar_desc("Poisoning")
                    self._update_progress_bar(
                        batch=batch_num, class_name=class_to_poison, remaining=samples_to_poson, n=len(batch))

                batch_num += 1
                self.handler.write_batch(trojaned_samples)

    def _validate_class_trojan_map(self,
                                   class_to_poison: Optional[str],
                                   amount_to_poison: float,
                                   class_trojan_map: Dict[str, Dict[int, float]]):
        if (class_to_poison is None or amount_to_poison is None) and (class_trojan_map is None):
            raise ValueError(
                "Either class_to_poison and amount_to_poison or class_trojan_map must be provided")

        class_names = self.handler.get_class_names()
        num_trojans = self.noise_generator.get_num_trojans()

        if class_trojan_map and amount_to_poison:
            if amount_to_poison <= 0 or amount_to_poison > 1:
                raise ValueError("amount_to_poison must be between 0 and 1")

            if class_to_poison not in class_names:
                raise ValueError(
                    f"class_to_poison {class_to_poison} not in dataset classes {class_names}")

            class_trojan_map = {
                class_to_poison: {0: amount_to_poison}
            }

        else:
            if not isinstance(class_trojan_map, dict):
                raise TypeError(
                    "class_trojan_map must be a dictionary of {class_names: {trojan_id: portion_to_poison}}")

            for class_name, poisoning_info in class_trojan_map.items():
                if not isinstance(class_name, str):
                    raise TypeError("class names must be strings")
                if not isinstance(poisoning_info, dict):
                    raise TypeError(
                        f"poisoning info for class {class_name} must be a dictionary")

                for trojan_idx, poisoning_proportion in poisoning_info.items():
                    if not isinstance(trojan_idx, int):
                        raise TypeError(
                            f"trojan id for class {class_name} must be an integer")
                    if not isinstance(poisoning_proportion, (float, int)):
                        raise TypeError(
                            f"poisoning proportion for class {class_name} must be a float")

            for class_name, poisoning_info in class_trojan_map.items():
                if class_name not in class_names:
                    raise ValueError(
                        f"class name {class_name} not in dataset classes {class_names}")

                for trojan_idx, poisoning_proportion in poisoning_info.items():
                    if min(poisoning_info.keys()) < 0 or max(poisoning_info.keys()) >= num_trojans:
                        raise ValueError(
                            f"trojan ids must be between 0 and {num_trojans-1}. Either correct the trojan_ids or increase the number of trojans in the noise generator")

                    if poisoning_proportion < 0 or poisoning_proportion > 1:
                        raise ValueError(
                            f"poisoning proportions must be between 0 and 1")

                    if sum(poisoning_info.values()) > 1.0:
                        raise ValueError(
                            f"Total poisoning proportion for class {class_name} cannot exceed 1.0")

        return class_trojan_map
