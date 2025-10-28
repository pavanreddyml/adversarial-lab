from PIL import Image
import numpy as np
import random
from typing import Any, List, Tuple, Optional

from adversarial_lab.core.noise_generators import NoiseGenerator


class TrojanImageNoiseGenerator(NoiseGenerator):
    def __init__(
        self,
        trojans: List[str | np.ndarray],
        size: Tuple[int, int] | int = (32, 32),
        position: Tuple[float, float] | Tuple[Tuple[float, float], Tuple[float, float]] = (10.0, 10.0),
        rotation: Tuple[float, float] | float = (0.0, 0.0),
        alpha: float | Tuple[float, float] = 1.0,
        keep_aspect_ratio: bool = True,
        fit_to_size: bool = True,
        coerce_out_of_bound: bool = True,
        
    ):
        if len(trojans) == 0:
            raise ValueError("At least one trojan must be provided.")

        # Load / normalize input trojans to (RGB uint8, ALPHA float32[0..1, HxWx1])
        if all(isinstance(t, np.ndarray) for t in trojans):
            trojan_arrs = trojans
        elif all(isinstance(t, str) for t in trojans):
            if not all(t.lower().endswith((".png", ".jpg", ".jpeg")) for t in trojans):
                raise ValueError("Trojans supported formats are .png, .jpg and .jpeg")
            trojan_arrs = [np.array(Image.open(t)) for t in trojans]
        else:
            raise ValueError("Trojans must be either all file paths or all numpy arrays.")

        self.trojans: List[np.ndarray] = []        # RGB uint8
        self.trojan_alphas: List[np.ndarray] = []  # HxWx1 float32 in [0,1]
        for tr in trojan_arrs:
            rgb, a = self._process_trojans(tr)
            self.trojans.append(rgb)
            self.trojan_alphas.append(a)

        # Process and validate parameters
        self.size = self._process_size(size)
        self.position = self._process_position(position)
        self.rotation = self._process_rotation(rotation)
        self.alpha = self._process_alpha(alpha)

        self.keep_aspect_ratio = keep_aspect_ratio
        self.fit_to_size = fit_to_size
        self.coerce_out_of_bound = coerce_out_of_bound

        self._RESIZE = getattr(Image, "LANCZOS", Image.BICUBIC)
        self._ROTATE = Image.BICUBIC

    # ----------------------------
    # Public API
    # ----------------------------
    def apply_noise(self, 
                    sample: np.ndarray, 
                    trojan_id: int, 
                    size: Optional[Tuple[int, int]] = None,
                    position: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                    rotation: Optional[Tuple[float, float]] = None,
                    alpha: Optional[Tuple[float, float]] = None,
                    *args, 
                    **kwargs) -> Any:
        # Fetch raw trojan components
        trojan_rgb = self.trojans[trojan_id]
        trojan_a = self.trojan_alphas[trojan_id]  # HxWx1 float [0,1]

        # 1) Match channels to sample
        trojan_rgb = self._match_channels_to_sample(trojan_rgb, self._shape_info(sample)[2])

        # 2) Resize
        size = self._process_size(size) if size is not None else self.size
        size = (random.randint(*size[0]), random.randint(*size[1]))
        trojan_rgb, trojan_a = self._resize(trojan_rgb, trojan_a, size=size)

        # 3) Rotate
        rotation = self._process_rotation(rotation) if rotation is not None else self.rotation
        angle = random.uniform(*rotation)
        trojan_rgb, trojan_a = self._rotate(trojan_rgb, trojan_a, angle=angle)

        # 4) Position
        position = self._process_position(position) if position is not None else self.position
        print(position)
        position = (random.uniform(*position[0]), random.uniform(*position[1]))
        print(position)
        x, y = self._get_absolute_xy(sample.shape, trojan_rgb.shape, position, self.coerce_out_of_bound)
        print(x, y)
        # 5) Overlay
        alpha = self._process_alpha(alpha) if alpha is not None else self.alpha
        alpha = random.uniform(*alpha)
        out = self._overlay(sample, trojan_rgb, trojan_a, x, y, alpha, self.coerce_out_of_bound)

        self._current_alpha = None
        return out

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _resize(self, 
                trojan_rgb: np.ndarray, 
                trojan_a: np.ndarray,
                size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        target_w, target_h = size

        rgb_img = self._np_to_pil(trojan_rgb)
        a_img = Image.fromarray((np.clip(trojan_a.squeeze(-1), 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

        tw, th = rgb_img.size
        if self.keep_aspect_ratio:
            if (tw <= target_w and th <= target_h):
                scale = max(target_w / tw, target_h / th) if self.fit_to_size else 1.0
            else:
                scale = min(target_w / tw, target_h / th)
            new_w = max(1, int(round(tw * scale)))
            new_h = max(1, int(round(th * scale)))
        else:
            new_w, new_h = target_w, target_h

        rgb_resized = rgb_img.resize((new_w, new_h), resample=self._RESIZE)
        a_resized = a_img.resize((new_w, new_h), resample=Image.BILINEAR)

        rgb_np = self._pil_to_np_matching_sample(rgb_resized, trojan_rgb.shape[2])
        a_np = (np.asarray(a_resized, dtype=np.float32) / 255.0)[..., None]
        return rgb_np, a_np

    def _rotate(self, 
                trojan_rgb: np.ndarray, 
                trojan_a: np.ndarray, 
                angle: float) -> Tuple[np.ndarray, np.ndarray]:

        rgb_img = self._np_to_pil(trojan_rgb)
        a_img = Image.fromarray((np.clip(trojan_a.squeeze(-1), 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

        # Fill for RGB depends on mode; for grayscale use median, else median RGB
        fill_rgb = 0 if rgb_img.mode == "L" else self._estimate_border_fillcolor(rgb_img)

        try:
            rgb_rot = rgb_img.rotate(angle=angle, resample=self._ROTATE, expand=True, fillcolor=fill_rgb)
        except TypeError:
            rgb_rot = rgb_img.rotate(angle=angle, resample=self._ROTATE, expand=True)

        try:
            a_rot = a_img.rotate(angle=angle, resample=Image.BILINEAR, expand=True, fillcolor=0)
        except TypeError:
            a_rot = a_img.rotate(angle=angle, resample=Image.BILINEAR, expand=True)

        rgb_np = self._pil_to_np_matching_sample(rgb_rot, trojan_rgb.shape[2])
        a_np = (np.asarray(a_rot, dtype=np.float32) / 255.0)[..., None]
        return rgb_np, a_np

    def _get_absolute_xy(
        self,
        sample_shape: Tuple[int, ...],
        trojan_shape: Tuple[int, ...],
        pos_percent: Tuple[float, float],
        coerce: bool,
    ) -> Tuple[int, int]:
        sh, sw, _ = self._shape_info_shape(sample_shape)
        th, tw = trojan_shape[0], trojan_shape[1]

        px = float(pos_percent[0])
        py = float(pos_percent[1])
        # Clamp to [0,1]
        px = 0.0 if px < 0.0 else 1.0 if px > 1.0 else px
        py = 0.0 if py < 0.0 else 1.0 if py > 1.0 else py

        # Available room so the trojan stays within bounds
        avail_w = max(sw - tw, 0)
        avail_h = max(sh - th, 0)

        # x: left padding; y: derived from bottom padding
        x = int(round(px * avail_w))
        y = int(round((1.0 - py) * avail_h))  # py=0 -> bottom; py=1 -> top

        if coerce:
            x = max(0, min(x, sw - tw))
            y = max(0, min(y, sh - th))

        return x, y

    def _overlay(
        self,
        sample_np: np.ndarray,
        trojan_rgb: np.ndarray,
        trojan_a: np.ndarray,
        x: int,
        y: int,
        alpha: float,
        coerce: bool,
    ) -> np.ndarray:
        sh, sw, sc = self._shape_info(sample_np)
        # Work on copy; normalize sample to uint8
        if sample_np.dtype != np.uint8:
            base_full = np.clip(sample_np, 0, 255).astype(np.uint8).copy()
        else:
            base_full = sample_np.copy()

        # If RGBA, operate on RGB and reattach alpha=255
        if sc == 4:
            base = base_full[:, :, :3]
        elif sc == 3:
            base = base_full
        elif sc == 1:
            base = base_full
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

        th, tw = trojan_rgb.shape[0], trojan_rgb.shape[1]

        # Compute overlap with canvas; crop trojan if needed
        x0 = 0 if not coerce else x
        y0 = 0 if not coerce else y
        if not coerce:
            x0 = max(0, x)
            y0 = max(0, y)

        sx0 = max(0, x)
        sy0 = max(0, y)
        sx1 = min(sw, x + tw)
        sy1 = min(sh, y + th)

        if sx0 >= sx1 or sy0 >= sy1:
            # No overlap
            if sc == 4:
                a_ch = np.full((sh, sw, 1), 255, dtype=np.uint8)
                return np.concatenate([base, a_ch], axis=2)
            return base

        tx0 = sx0 - x
        ty0 = sy0 - y
        tx1 = tx0 + (sx1 - sx0)
        ty1 = ty0 + (sy1 - sy0)

        # Ensure channel match for trojan vs base
        if base.ndim == 2:
            # Grayscale
            t_rgb = trojan_rgb[ty0:ty1, tx0:tx1]
            t_a = trojan_a[ty0:ty1, tx0:tx1, 0] * alpha  # HxW
            b = base[sy0:sy1, sx0:sx1].astype(np.float32)
            t = t_rgb.astype(np.float32)
            a_map = np.clip(t_a, 0.0, 1.0).astype(np.float32)
            out = (a_map * t + (1.0 - a_map) * b).round().astype(np.uint8)
            base[sy0:sy1, sx0:sx1] = out
            if sc == 4:
                a_ch = np.full((sh, sw, 1), 255, dtype=np.uint8)
                return np.concatenate([base, a_ch], axis=2)
            return base

        else:
            # RGB
            t_rgb = trojan_rgb[ty0:ty1, tx0:tx1, :]
            t_a = trojan_a[ty0:ty1, tx0:tx1, 0] * alpha  # HxW
            b = base[sy0:sy1, sx0:sx1, :].astype(np.float32)
            t = t_rgb.astype(np.float32)
            a_map = np.clip(t_a, 0.0, 1.0).astype(np.float32)[..., None]  # HxWx1
            out = (a_map * t + (1.0 - a_map) * b).round().astype(np.uint8)
            base[sy0:sy1, sx0:sx1, :] = out
            if sc == 4:
                a_ch = np.full((sh, sw, 1), 255, dtype=np.uint8)
                return np.concatenate([base, a_ch], axis=2)
            return base

    # ----------------------------
    # Utilities
    # ----------------------------
    def _process_trojans(self, trojan_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if trojan_arr.dtype != np.uint8:
            trojan_arr = np.clip(trojan_arr, 0, 255).astype(np.uint8)

        if trojan_arr.ndim == 2:
            rgb = np.repeat(trojan_arr[..., None], 3, axis=2)
            alpha = np.ones(trojan_arr.shape[:2] + (1,), dtype=np.float32)
        elif trojan_arr.ndim == 3 and trojan_arr.shape[2] == 1:
            rgb = np.repeat(trojan_arr, 3, axis=2)
            alpha = np.ones(trojan_arr.shape[:2] + (1,), dtype=np.float32)
        elif trojan_arr.ndim == 3 and trojan_arr.shape[2] == 3:
            rgb = trojan_arr
            alpha = np.ones(trojan_arr.shape[:2] + (1,), dtype=np.float32)
        elif trojan_arr.ndim == 3 and trojan_arr.shape[2] == 4:
            rgba = trojan_arr
            alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
            rgb = rgba[:, :, :3].astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image shape: {trojan_arr.shape}")
        return rgb, alpha
    
    def _process_position(self, pos):
        """Accept (x, y) or ((x1, x2), (y1, y2)) with values in [0, 1]."""
        if not (isinstance(pos, tuple) and len(pos) == 2):
            raise ValueError("Position must be (x, y) or ((x1, x2), (y1, y2))")

        # Single (x, y)
        if all(isinstance(p, (int, float)) for p in pos):
            x, y = float(pos[0]), float(pos[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Position values must be in [0, 1]")
            return ((x, x), (y, y))

        # Ranges ((x1, x2), (y1, y2))
        if all(isinstance(p, tuple) and len(p) == 2 for p in pos):
            (x1, x2), (y1, y2) = pos
            vals = [x1, x2, y1, y2]
            if not all(isinstance(v, (int, float)) for v in vals):
                raise ValueError("Position values must be numbers in [0, 1]")
            x1, x2, y1, y2 = map(float, (x1, x2, y1, y2))
            if not all(0.0 <= v <= 1.0 for v in (x1, x2, y1, y2)):
                raise ValueError("Position values must be in [0, 1]")
            if x1 > x2 or y1 > y2:
                raise ValueError("Minimum position cannot be greater than maximum")
            return ((x1, x2), (y1, y2))

        raise ValueError("Position must be (x, y) or ((x1, x2), (y1, y2))")

    def _process_rotation(self, rot):
        if isinstance(rot, (int, float)):
            r = float(rot)
            if not (-360.0 <= r <= 360.0):
                raise ValueError("Rotation angle must be between -360 and 360 degrees")
            return (r, r)
        elif isinstance(rot, tuple) and len(rot) == 2:
            r0, r1 = float(rot[0]), float(rot[1])
            if not (-360.0 <= r0 <= 360.0 and -360.0 <= r1 <= 360.0):
                raise ValueError("Rotation angles must be between -360 and 360 degrees")
            if r0 > r1:
                raise ValueError("Minimum rotation cannot be greater than maximum")
            return (r0, r1)
        else:
            raise ValueError("Rotation must be a number or (min_deg, max_deg) tuple")

    def _process_size(self, size):
        """Process size as single int, (width, height) or ((w1, w2), (h1, h2)) ranges."""
        if isinstance(size, int):
            if size <= 0:
                raise ValueError("Size must be positive")
            return ((size, size), (size, size))
        elif isinstance(size, tuple) and len(size) == 2:
            # Single (width, height) case
            if all(isinstance(s, int) for s in size):
                width, height = size
                if width <= 0 or height <= 0:
                    raise ValueError("Size values must be positive")
                return ((width, width), (height, height))
            
            # Range ((w1, w2), (h1, h2)) case
            if all(isinstance(s, tuple) and len(s) == 2 for s in size):
                (w1, w2), (h1, h2) = size
                values = [w1, w2, h1, h2]
                if not all(isinstance(v, int) for v in values):
                    raise ValueError("Size values must be integers")
                if not all(v > 0 for v in values):
                    raise ValueError("Size values must be positive")
                if w1 > w2 or h1 > h2:
                    raise ValueError("Minimum size cannot be greater than maximum")
                return ((w1, w2), (h1, h2))
            
            raise ValueError("Size must be (width, height) or ((w1, w2), (h1, h2))")
        else:
            raise ValueError("Size must be an integer, (width, height) or ((w1, w2), (h1, h2))")

    def _process_alpha(self, alpha):
        if isinstance(alpha, (int, float)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("Alpha must be between 0.0 and 1.0")
            return (alpha_val, alpha_val)
        elif isinstance(alpha, tuple) and len(alpha) == 2:
            min_alpha, max_alpha = float(alpha[0]), float(alpha[1])
            if not (0.0 <= min_alpha <= 1.0 and 0.0 <= max_alpha <= 1.0):
                raise ValueError("Alpha values must be between 0.0 and 1.0")
            if min_alpha > max_alpha:
                raise ValueError("Minimum alpha cannot be greater than maximum")
            return (min_alpha, max_alpha)
        else:
            raise ValueError("Alpha must be a number or (min_alpha, max_alpha) tuple")

    def _match_channels_to_sample(self, trojan_rgb: np.ndarray, sample_c: int) -> np.ndarray:
        if sample_c == 1:
            # Convert RGB->L
            img = Image.fromarray(trojan_rgb, mode="RGB") if trojan_rgb.ndim == 3 and trojan_rgb.shape[2] == 3 else Image.fromarray(trojan_rgb, mode="L")
            if img.mode != "L":
                img = img.convert("L")
            return np.asarray(img, dtype=np.uint8)
        elif sample_c in (3, 4):
            if trojan_rgb.ndim == 2:
                img = Image.fromarray(trojan_rgb, mode="L").convert("RGB")
                return np.asarray(img, dtype=np.uint8)
            if trojan_rgb.ndim == 3 and trojan_rgb.shape[2] == 3:
                return trojan_rgb
            if trojan_rgb.ndim == 3 and trojan_rgb.shape[2] == 1:
                img = Image.fromarray(trojan_rgb[:, :, 0], mode="L").convert("RGB")
                return np.asarray(img, dtype=np.uint8)
            raise ValueError("Unsupported trojan shape for channel matching.")
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

    @staticmethod
    def _shape_info(arr: np.ndarray) -> Tuple[int, int, int]:
        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
        elif arr.ndim == 3:
            h, w, c = arr.shape
        else:
            raise ValueError("sample must be a 2D (grayscale) or 3D (H,W,C) array.")
        return h, w, c

    @staticmethod
    def _shape_info_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        if len(shape) == 2:
            h, w = shape
            c = 1
        elif len(shape) == 3:
            h, w, c = shape
        else:
            raise ValueError("Shape must be length 2 or 3.")
        return h, w, c

    @staticmethod
    def _np_to_pil(arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L")
        if arr.ndim == 3:
            c = arr.shape[2]
            if c == 1:
                return Image.fromarray(arr[:, :, 0].astype(np.uint8), mode="L")
            elif c == 3:
                return Image.fromarray(arr.astype(np.uint8), mode="RGB")
            elif c == 4:
                return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
        raise ValueError("trojan must be 2D or 3D uint8 array with 1/3/4 channels.")

    @staticmethod
    def _pil_to_np_matching_sample(img: Image.Image, sample_c: int) -> np.ndarray:
        if sample_c == 1:
            if img.mode != "L":
                img = img.convert("L")
            return np.asarray(img, dtype=np.uint8)
        elif sample_c == 3:
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        elif sample_c == 4:
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

    @staticmethod
    def _estimate_border_fillcolor(img: Image.Image):
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        arr = np.asarray(img)
        if img.mode == "L":
            top = arr[0, :]
            bottom = arr[-1, :]
            left = arr[:, 0]
            right = arr[:, -1]
            border = np.concatenate([top, bottom, left, right], axis=0)
            val = int(np.median(border))
            return val
        else:
            top = arr[0, :, :]
            bottom = arr[-1, :, :]
            left = arr[:, 0, :]
            right = arr[:, -1, :]
            border = np.concatenate([top, bottom, left, right], axis=0)
            med = np.median(border, axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))

    def get_num_trojans(self) -> int:
        return len(self.trojans)
