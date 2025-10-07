# jpeg_compression.py
from io import BytesIO
from typing import Tuple, Optional

import numpy as np
from PIL import Image

from . import BasePreprocessingDefense


class JPEGCompression(BasePreprocessingDefense):
    def __init__(
        self,
        quality: int = 75,
        subsampling: Optional[int] = "keep",
        keep_size: bool = True,
        color_mode: Optional[str] = None,
    ):
        self.quality = int(np.clip(quality, 1, 100))
        self.subsampling = subsampling
        self.keep_size = keep_size
        self.color_mode = color_mode

    def _to_pil(self, arr: np.ndarray) -> Tuple[Image.Image, Tuple[int, int]]:
        if arr.ndim == 2:
            mode = "L"
            data = arr
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            mode = "L" if arr.shape[2] == 1 else "RGB"
            data = arr.squeeze()
        else:
            raise ValueError("JPEGCompression expects HxW or HxWx[1|3] array.")

        orig_dtype = arr.dtype
        if np.issubdtype(orig_dtype, np.floating):
            x = np.clip(data, 0.0, 1.0)
            x = (x * 255.0 + 0.5).astype(np.uint8)
        else:
            x = data.astype(np.uint8)
        img = Image.fromarray(x, mode=mode if self.color_mode is None else self.color_mode)
        return img, data.shape[:2]

    def _from_pil(self, img: Image.Image, target_size: Tuple[int, int], like: np.ndarray) -> np.ndarray:
        if self.keep_size and img.size[::-1] != target_size:
            img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.asarray(img)
        if like.ndim == 3 and like.shape[2] == 3 and arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if like.ndim == 3 and like.shape[2] == 1 and arr.ndim == 2:
            arr = arr[..., None]
        if np.issubdtype(like.dtype, np.floating):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(like.dtype)
        return arr

    def run(self, data: np.ndarray) -> np.ndarray:
        img, target_size = self._to_pil(data)
        buf = BytesIO()
        save_kwargs = {"format": "JPEG", "quality": self.quality}
        if self.subsampling != "keep":
            save_kwargs["subsampling"] = self.subsampling
        img.save(buf, **save_kwargs)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return self._from_pil(out, target_size, like=data)