from typing import Tuple, Optional

import numpy as np
from PIL import Image

from . import BasePreprocessingDefense


class RandomResizePad(BasePreprocessingDefense):
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        interpolation=getattr(Image, "Resampling", Image).BILINEAR,  # Pillow ≥10 compat
    ):
        assert 0 < scale_range[0] <= scale_range[1]
        assert pad_mode in ("constant", "edge")
        self.scale_range = scale_range
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.rng = rng or np.random.default_rng()
        self.interp = interpolation

    def _to_uint8(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        if np.issubdtype(x.dtype, np.floating):
            return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), True
        return x.astype(np.uint8), False

    def _from_uint8(self, y: np.ndarray, was_float: bool, like: np.ndarray) -> np.ndarray:
        if was_float:
            y = y.astype(np.float32) / 255.0
            return y
        return y.astype(like.dtype)

    def _resize(self, arr: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            mode = "L" if arr.shape[2] == 1 else "RGB"
            img = Image.fromarray(arr.squeeze() if arr.shape[2] == 1 else arr, mode=mode)
        else:
            raise ValueError("RandomResizePad expects HxW or HxWx[1|3] array.")
        img = img.resize((new_hw[1], new_hw[0]), self.interp)
        out = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 1 and out.ndim == 2:
            out = out[..., None]
        return out

    def run(self, data: np.ndarray) -> np.ndarray:
        H, W = data.shape[:2]
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        new_h, new_w = max(1, int(round(H * scale))), max(1, int(round(W * scale)))

        x_uint8, was_float = self._to_uint8(data)
        resized = self._resize(x_uint8, (new_h, new_w))

        out_shape = (H, W) if data.ndim == 2 else (H, W, data.shape[2])
        out = np.zeros(out_shape, dtype=resized.dtype)

        if new_h >= H and new_w >= W:
            top = (new_h - H) // 2
            left = (new_w - W) // 2
            out = resized[top : top + H, left : left + W, ...]
        elif self.pad_mode == "edge":
            pad_h_top = max(0, (H - new_h) // 2)
            pad_h_bot = max(0, H - new_h - pad_h_top)
            pad_w_left = max(0, (W - new_w) // 2)
            pad_w_right = max(0, W - new_w - pad_w_left)
            pad_spec = (
                (pad_h_top, pad_h_bot),
                (pad_w_left, pad_w_right),
            )
            if data.ndim == 3:
                pad_spec += ((0, 0),)
            padded = np.pad(resized, pad_spec, mode="edge")
            out = padded[:H, :W, ...] if data.ndim == 3 else padded[:H, :W]
        else:
            fill_val = int(np.clip(self.pad_value * (255 if was_float else 1), 0, 255))
            out.fill(fill_val)

            dst_h = min(H, new_h)
            dst_w = min(W, new_w)
            dst_top = max(0, (H - new_h) // 2)
            dst_left = max(0, (W - new_w) // 2)
            dst_bottom = dst_top + dst_h
            dst_right = dst_left + dst_w

            src_top = max(0, (new_h - H) // 2)
            src_left = max(0, (new_w - W) // 2)
            src_bottom = src_top + dst_h
            src_right = src_left + dst_w

            if data.ndim == 2:
                out[dst_top:dst_bottom, dst_left:dst_right] = resized[src_top:src_bottom, src_left:src_right]
            else:
                out[dst_top:dst_bottom, dst_left:dst_right, :] = resized[src_top:src_bottom, src_left:src_right, :]

        return self._from_uint8(out, was_float, like=data)
