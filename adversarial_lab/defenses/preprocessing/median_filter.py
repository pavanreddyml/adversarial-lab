from typing import Tuple

import numpy as np

from . import BasePreprocessingDefense

# Optional backends
try:
    from scipy.ndimage import median_filter as _scipy_median
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class MedianFilter(BasePreprocessingDefense):
    def __init__(self, ksize: Tuple[int, int] | int = 3):
        self.ksize = ksize

    def _median_naive(self, x: np.ndarray, k: Tuple[int, int]) -> np.ndarray:
        kh, kw = k
        pad_h, pad_w = kh // 2, kw // 2
        if x.ndim == 2:
            xpad = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
            out = np.empty_like(x)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    window = xpad[i : i + kh, j : j + kw]
                    out[i, j] = np.median(window)
            return out
        elif x.ndim == 3:
            out = np.empty_like(x)
            for c in range(x.shape[2]):
                out[..., c] = self._median_naive(x[..., c], (kh, kw))
            return out
        else:
            raise ValueError("MedianFilter expects 2D or 3D array.")

    def run(self, data: np.ndarray) -> np.ndarray:
        k = self.ksize if isinstance(self.ksize, tuple) else (self.ksize, self.ksize)
        if _HAS_SCIPY:
            if data.ndim == 2:
                return _scipy_median(data, size=k, mode="nearest")
            elif data.ndim == 3:
                # Apply per-channel to avoid mixing channels
                out = np.empty_like(data)
                for c in range(data.shape[2]):
                    out[..., c] = _scipy_median(data[..., c], size=k, mode="nearest")
                return out
            else:
                raise ValueError("MedianFilter expects 2D or 3D array.")
        else:
            return self._median_naive(data, k)
