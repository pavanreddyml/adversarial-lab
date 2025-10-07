# total_variation_minimization.py
from typing import Optional

import numpy as np

from . import BasePreprocessingDefense

# Optional backend
try:
    from skimage.restoration import denoise_tv_chambolle as _tv_denoise
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


class TotalVariationMinimization(BasePreprocessingDefense):
    def __init__(self, weight: float = 0.1, eps: float = 2e-4, n_iter_max: int = 200):
        self.weight = float(weight)
        self.eps = float(eps)
        self.n_iter_max = int(n_iter_max)

    def _fallback(self, x: np.ndarray, iters: int = 20, tau: float = 0.125) -> np.ndarray:
        u = x.astype(np.float32, copy=True)
        for _ in range(iters):
            ux = np.roll(u, -1, axis=1) - u
            uy = np.roll(u, -1, axis=0) - u
            nx = ux / (np.sqrt(ux * ux + 1e-6))
            ny = uy / (np.sqrt(uy * uy + 1e-6))
            div = (nx - np.roll(nx, 1, axis=1)) + (ny - np.roll(ny, 1, axis=0))
            u = u + tau * (div - (u - x) / max(self.weight, 1e-6))
        return u.astype(np.float32)

    def run(self, data: np.ndarray) -> np.ndarray:
        if _HAS_SKIMAGE:
            if data.ndim == 2:
                out = _tv_denoise(data, weight=self.weight, eps=self.eps, n_iter_max=self.n_iter_max, channel_axis=None)
            elif data.ndim == 3:
                out = _tv_denoise(data, weight=self.weight, eps=self.eps, n_iter_max=self.n_iter_max, channel_axis=-1)
            else:
                raise ValueError("TotalVariationMinimization expects 2D or 3D array.")
            return out.astype(np.float32) if np.issubdtype(data.dtype, np.floating) else np.clip(out, 0, 255).astype(data.dtype)
        else:
            if data.ndim == 3:
                chans = [self._fallback(data[..., c]) for c in range(data.shape[2])]
                out = np.stack(chans, axis=-1)
            elif data.ndim == 2:
                out = self._fallback(data)
            else:
                raise ValueError("TotalVariationMinimization expects 2D or 3D array.")
            if np.issubdtype(data.dtype, np.floating):
                return np.clip(out, 0.0, 1.0).astype(np.float32)
            return np.clip(out, 0, 255).astype(data.dtype)
