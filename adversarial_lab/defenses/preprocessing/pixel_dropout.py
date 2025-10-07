# pixel_dropout.py
from typing import Optional, Tuple

import numpy as np

from . import BasePreprocessingDefense


class PixelDropout(BasePreprocessingDefense):
    def __init__(
        self,
        p: float = 0.05,
        fill_value: Optional[float] = None,
        per_channel: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.fill_value = fill_value
        self.per_channel = per_channel
        self.rng = rng or np.random.default_rng()

    def _compute_fill(self, x: np.ndarray):
        if self.fill_value is not None:
            return self.fill_value
        if x.ndim == 3 and x.shape[2] > 1:
            return x.mean(axis=(0, 1), keepdims=False)
        return float(x.mean())

    def run(self, data: np.ndarray) -> np.ndarray:
        x = data.copy()
        fill = self._compute_fill(x)

        if x.ndim == 2:
            mask = self.rng.random(x.shape) < self.p
            x[mask] = fill if np.isscalar(fill) else float(fill)
            return x

        if x.ndim == 3:
            H, W, C = x.shape
            if self.per_channel:
                mask = self.rng.random((H, W, C)) < self.p
            else:
                mask2d = self.rng.random((H, W)) < self.p
                mask = np.repeat(mask2d[..., None], C, axis=2)
            if np.isscalar(fill):
                x[mask] = fill
            else:
                # broadcast per-channel fill
                x[mask] = np.take(fill, np.where(mask)[2])
            return x

        raise ValueError("PixelDropout expects 2D or 3D array.")