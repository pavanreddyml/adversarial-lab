# randomized_smoothing.py
from typing import Optional

import numpy as np

from . import BasePreprocessingDefense


class RandomizedSmoothing(BasePreprocessingDefense):
    def __init__(
        self,
        sigma: float = 0.25 / 255.0,
        num_samples: int = 8,
        clip: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.sigma = float(sigma)
        self.num_samples = int(num_samples)
        self.clip = clip
        self.rng = rng or np.random.default_rng()

    def _bounds(self, data: np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            return 0.0, 1.0
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            return info.min, info.max
        return None

    def run(self, data: np.ndarray) -> np.ndarray:
        if self.num_samples <= 1:
            return data

        if np.issubdtype(data.dtype, np.integer):
            scale = 1.0
            base = data.astype(np.float32)
        else:
            scale = 1.0
            base = data.astype(np.float32)

        acc = np.zeros_like(base, dtype=np.float32)
        for _ in range(self.num_samples):
            noise = self.rng.normal(0.0, self.sigma * scale, size=base.shape).astype(np.float32)
            acc += base + noise

        out = acc / float(self.num_samples)

        lo, hi = self._bounds(data)
        if self.clip and lo is not None:
            out = np.clip(out, lo, hi)

        if np.issubdtype(data.dtype, np.integer):
            out = np.rint(out).astype(data.dtype)
        else:
            out = out.astype(np.float32)
        return out