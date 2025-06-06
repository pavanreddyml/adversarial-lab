from typing import Dict, List, Literal, Union
from . import Tracker
import numpy as np

TrackedStat = Literal["mean", "median", "std", "min", "max", "var", "p25", "p75", "p_custom_x", "iqr", "skew", "kurtosis"]


class NoiseStatisticsTracker(Tracker):
    _columns = {
        "epoch_noise_stats": "json",
        "epoch_noise_stats_by_batch": "json"
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 tracked_stats: List[Union[TrackedStat, str]] = ["mean", "median", "std", "min", "max", "var", "p25", "p75", "iqr", "skew", "kurtosis"],
                 round_to: int = 8
                 ) -> None:
        self.tracked_stats = tracked_stats
        self.round_to = round_to
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)
        self.reset_values()

    def post_batch(self, *args, **kwargs) -> None:
        if not self.track_batch:
            return

        noise = kwargs.get("noise", None)
        if noise is None:
            return

        noise = noise.flatten()
        stats = self._get_stats(noise)
        self.epoch_noise_stats_by_batch.append(stats)

    def post_epoch(self, *args, **kwargs) -> None:
        if not self.track_epoch:
            return

        noise = kwargs.get("noise", None)
        if noise is None:
            return

        noise = noise.flatten()
        stats = self._get_stats(noise)
        self.epoch_noise_stats = stats

    def _get_stats(self, noise: np.ndarray) -> Dict[str, float]:
        stats = {}
        mean = np.mean(noise)
        std = np.std(noise)
        for stat in self.tracked_stats:
            try:
                if stat == "mean":
                    stats["mean"] = round(float(mean), self.round_to)
                elif stat == "median":
                    stats["median"] = round(float(np.median(noise)), self.round_to)
                elif stat == "std":
                    stats["std"] = round(float(std), self.round_to)
                elif stat == "min":
                    stats["min"] = round(float(np.min(noise)), self.round_to)
                elif stat == "max":
                    stats["max"] = round(float(np.max(noise)), self.round_to)
                elif stat == "var":
                    stats["var"] = round(float(np.var(noise)), self.round_to)
                elif stat == "p25":
                    stats["p25"] = round(float(np.percentile(noise, 25)), self.round_to)
                elif stat == "p75":
                    stats["p75"] = round(float(np.percentile(noise, 75)), self.round_to)
                elif stat == "iqr":
                    q1 = np.percentile(noise, 25)
                    q3 = np.percentile(noise, 75)
                    stats["iqr"] = round(float(q3 - q1), self.round_to)
                elif stat == "skew":
                    stats["skew"] = round(float(np.mean((noise - mean)**3) / std**3) if std != 0 else 0.0, self.round_to)
                elif stat == "kurtosis":
                    stats["kurtosis"] = round(float(np.mean((noise - mean)**4) / std**4 - 3) if std != 0 else -3.0, self.round_to)
                elif stat.startswith("p_custom_"):
                    val = float(stat.split("p_custom_")[1])
                    stats[stat] = round(float(np.percentile(noise, val)), self.round_to)
            except Exception:
                continue
        return stats

    def serialize(self) -> Dict:
        data = {}
        if self.track_epoch:
            data["epoch_noise_stats"] = self.epoch_noise_stats
        if self.track_batch:
            data["epoch_noise_stats_by_batch"] = self.epoch_noise_stats_by_batch
        return data

    def reset_values(self) -> None:
        self.epoch_noise_stats = {}
        self.epoch_noise_stats_by_batch: List[Dict[str, float]] = []
