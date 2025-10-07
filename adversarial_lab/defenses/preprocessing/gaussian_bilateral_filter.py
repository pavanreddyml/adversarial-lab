from typing import Optional, Tuple

import numpy as np

from . import BasePreprocessingDefense

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from skimage.restoration import denoise_bilateral as _sk_bilateral
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


class GaussianBilateralFilter(BasePreprocessingDefense):
    def __init__(
        self,
        mode: str = "gaussian",
        gaussian_sigma: float | Tuple[float, float] = 1.0,
        bilateral_diameter: Optional[int] = None,
        bilateral_sigma_color: float = 0.1,
        bilateral_sigma_space: float = 3.0,
        preserve_range: bool = True,
    ):
        assert mode in ("gaussian", "bilateral")
        self.mode = mode
        self.gaussian_sigma = gaussian_sigma
        self.bilateral_diameter = bilateral_diameter
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.preserve_range = preserve_range

    def _apply_gaussian(self, x: np.ndarray) -> np.ndarray:
        if not _HAS_SCIPY:
            return x
        if x.ndim == 2:
            return _gaussian_filter(x, sigma=self.gaussian_sigma, mode="nearest")
        elif x.ndim == 3:
            out = np.empty_like(x)
            for c in range(x.shape[2]):
                out[..., c] = _gaussian_filter(x[..., c], sigma=self.gaussian_sigma, mode="nearest")
            return out
        else:
            raise ValueError("Gaussian expects 2D or 3D array.")

    def _apply_bilateral(self, x: np.ndarray) -> np.ndarray:
        if _HAS_CV2:
            was_float = np.issubdtype(x.dtype, np.floating)
            if was_float:
                xcv = x.astype(np.float32)
                sigma_color = float(self.bilateral_sigma_color)
                sigma_space = float(self.bilateral_sigma_space)
            else:
                xcv = x
                sigma_color = float(self.bilateral_sigma_color * (255.0 if x.dtype == np.uint8 else 1.0))
                sigma_space = float(self.bilateral_sigma_space)

            d = self.bilateral_diameter if self.bilateral_diameter is not None else 0
            if x.ndim == 2:
                out = cv2.bilateralFilter(xcv, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            elif x.ndim == 3:
                out = cv2.bilateralFilter(xcv, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            else:
                raise ValueError("Bilateral expects 2D or 3D array.")
            return out.astype(x.dtype) if not was_float else out.astype(np.float32)

        if _HAS_SKIMAGE:
            if x.ndim == 2:
                return _sk_bilateral(
                    x, sigma_color=self.bilateral_sigma_color, sigma_spatial=self.bilateral_sigma_space, preserve_range=self.preserve_range
                ).astype(x.dtype if not np.issubdtype(x.dtype, np.floating) else np.float32)
            elif x.ndim == 3:
                out = np.empty_like(x, dtype=(x.dtype if not np.issubdtype(x.dtype, np.floating) else np.float32))
                for c in range(x.shape[2]):
                    out[..., c] = _sk_bilateral(
                        x[..., c],
                        sigma_color=self.bilateral_sigma_color,
                        sigma_spatial=self.bilateral_sigma_space,
                        preserve_range=self.preserve_range,
                    )
                return out
            else:
                raise ValueError("Bilateral expects 2D or 3D array.")

        return x

    def run(self, data: np.ndarray) -> np.ndarray:
        if self.mode == "gaussian":
            return self._apply_gaussian(data)
        else:
            return self._apply_bilateral(data)