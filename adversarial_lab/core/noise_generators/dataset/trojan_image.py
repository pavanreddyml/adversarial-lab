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
        position: Tuple[float, float] = (10.0, 10.0),  # (x%, y%) with 0,0 bottom-left and 100,100 top-right
        rotation: Tuple[float, float] | float = (0.0, 0.0),
        keep_aspect_ratio: bool = True,
        fit_to_size: bool = True,
        coerce_out_of_bound: bool = True,
        alpha: float = 1.0,  # NEW: global alpha in [0,1]
    ):
        if all(isinstance(trojan, np.ndarray) for trojan in trojans):
            self.trojans = trojans
        elif all(isinstance(trojan, str) for trojan in trojans):
            if not all(
                trojan.lower().endswith((".png", ".jpg", ".jpeg")) for trojan in trojans
            ):
                raise ValueError("Trojans supported formats are .png, .jpg and .jpeg")
            self.trojans = [np.array(Image.open(t).convert("RGBA")) for t in trojans]
        else:
            raise ValueError("Trojans must be either all file paths or all numpy arrays.")

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple) and len(size) == 2:
            self.size = size
        else:
            raise ValueError("Size must be either an int or a tuple of two ints.")

        if not (
            isinstance(position, tuple)
            and len(position) == 2
            and all(0.0 <= p <= 100.0 for p in position)
        ):
            raise ValueError("Position must be a tuple of two floats between 0.0 and 100.0 (percent).")
        self.position = position

        if isinstance(rotation, float):
            if not (0.0 <= rotation <= 1.0):
                raise ValueError("Rotation must be a float between 0.0 and 1.0.")
            self.rotation = (-rotation, rotation)
        elif isinstance(rotation, tuple) and len(rotation) == 2:
            if not all(0.0 <= r <= 1.0 for r in rotation):
                raise ValueError("Rotation values must be between 0.0 and 1.0.")
            self.rotation = rotation
        else:
            raise ValueError("Rotation must be either a float or a tuple of two floats.")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.global_alpha = alpha  # NEW

        self.keep_aspect_ratio = keep_aspect_ratio
        self.fit_to_size = fit_to_size
        self.coerce_out_of_bound = coerce_out_of_bound

        # Pillow resampling handles (compat with older/newer Pillow)
        self._RESIZE = getattr(Image, "LANCZOS", Image.BICUBIC)
        self._ROTATE = Image.BICUBIC

    def apply_noise(
        self, sample: np.ndarray, trojan_id: int, *args, alpha: Optional[float] = None, **kwargs
    ) -> Any:
        """
        Returns the sample with the trojan composited at the requested position.
        Position is (x%, y%) with 0,0 at bottom-left and 100,100 at top-right.
        Optional `alpha` overrides the global alpha for this call.
        """
        if alpha is None:
            alpha = self.global_alpha
        else:
            if not (0.0 <= alpha <= 1.0):
                raise ValueError("alpha must be in [0, 1].")

        raw_trojan = self.trojans[trojan_id]
        trojan = self._get_trojan_reshaped(sample, raw_trojan)[0]
        out = self._composite_with_position_percent(
            sample, trojan, self.position, self.coerce_out_of_bound, alpha_override=alpha
        )
        return out

    def _get_trojan_reshaped(self, sample: np.ndarray, trojan: np.ndarray) -> List[np.ndarray]:
        sample_h, sample_w, sample_c = self._shape_info(sample)
        target_w, target_h = self.size

        trojan_img = self._np_to_pil(trojan)
        trojan_img = self._convert_mode_to_match_sample(trojan_img, sample_c)

        tw, th = trojan_img.size

        if self.keep_aspect_ratio:
            if (tw <= target_w and th <= target_h):
                scale = max(target_w / tw, target_h / th) if self.fit_to_size else 1.0
            else:
                scale = min(target_w / tw, target_h / th)
            new_w = max(1, int(round(tw * scale)))
            new_h = max(1, int(round(th * scale)))
        else:
            new_w, new_h = target_w, target_h

        resized = trojan_img.resize((new_w, new_h), resample=self._RESIZE)

        min_r, max_r = self.rotation
        angle_deg = random.uniform(min_r, max_r) * 360.0
        rotated = resized.rotate(angle=angle_deg, resample=self._ROTATE, expand=True)

        trojan_np = self._pil_to_np_matching_sample(rotated, sample_c)
        return [trojan_np]

    # ----------------- helpers -----------------

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
                return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        raise ValueError("trojan must be 2D or 3D uint8 array with 1/3/4 channels.")

    @staticmethod
    def _pil_to_np_matching_sample(img: Image.Image, sample_c: int) -> np.ndarray:
        if sample_c == 1:
            if img.mode != "L":
                img = img.convert("L")
            out = np.asarray(img, dtype=np.uint8)
            return out
        elif sample_c == 3:
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        elif sample_c == 4:
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            return np.asarray(img, dtype=np.uint8)
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

    @staticmethod
    def _convert_mode_to_match_sample(img: Image.Image, sample_c: int) -> Image.Image:
        if sample_c == 1 and img.mode != "L":
            return img.convert("L")
        if sample_c == 3 and img.mode != "RGB":
            return img.convert("RGB")
        if sample_c == 4 and img.mode != "RGBA":
            base = img.convert("RGBA")
            if img.mode == "RGB":
                r, g, b = base.split()[:3]
                a = Image.new("L", base.size, 255)
                base = Image.merge("RGBA", (r, g, b, a))
            return base
        return img

    # ----------------- NEW: compositing with percent position + alpha -----------------

    def _composite_with_position_percent(
        self,
        sample_np: np.ndarray,
        trojan_np: np.ndarray,
        pos_percent: Tuple[float, float],
        coerce: bool,
        alpha_override: float,
    ) -> np.ndarray:
        """
        Composite trojan onto sample.
        pos_percent: (x%, y%) with 0,0 at bottom-left and 100,100 at top-right.
        Anchor is top-left of the trojan at that (x%, y%) position.
        If `coerce` is True, the trojan is shifted so it is fully inside the image.
        If `coerce` is False, the trojan is cropped to the visible region.
        alpha_override in [0,1] scales the trojan's alpha globally.
        The returned array has the same channel count as `sample_np`.
        """
        sh, sw, sc = self._shape_info(sample_np)

        # Convert arrays to PIL for compositing in RGBA space
        if sc == 1:
            base = Image.fromarray(sample_np.astype(np.uint8), mode="L").convert("RGBA")
        elif sc == 3:
            base = Image.fromarray(sample_np.astype(np.uint8), mode="RGB").convert("RGBA")
        elif sc == 4:
            base = Image.fromarray(sample_np.astype(np.uint8), mode="RGBA")
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

        trojan_img = self._np_to_pil(trojan_np)
        # Ensure overlay is RGBA to leverage alpha (synthetic opaque if none)
        if trojan_img.mode != "RGBA":
            trojan_img = trojan_img.convert("RGBA")

        # Apply global alpha by scaling the A channel
        if alpha_override < 1.0:
            r, g, b, a = trojan_img.split()
            a = a.point(lambda v: int(round(v * alpha_override)))
            trojan_img = Image.merge("RGBA", (r, g, b, a))

        fw, fh = trojan_img.size

        # Position: (x%, y%) in [0,100], origin bottom-left.
        px_pct, py_pct = pos_percent
        x = int(round((px_pct / 100.0) * sw))
        y_from_bottom = int(round((py_pct / 100.0) * sh))
        y = sh - y_from_bottom - fh  # convert to top-left y (PIL coordinates)

        if coerce:
            # Shift to keep fully inside
            x = max(0, min(x, sw - fw))
            y = max(0, min(y, sh - fh))
            base.paste(trojan_img, (x, y), trojan_img)
        else:
            # Allow out-of-bounds; crop to visible intersection
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(sw, x + fw), min(sh, y + fh)

            if x0 < x1 and y0 < y1:
                crop_left   = x0 - x
                crop_top    = y0 - y
                crop_right  = crop_left + (x1 - x0)
                crop_bottom = crop_top + (y1 - y0)
                trojan_cropped = trojan_img.crop((crop_left, crop_top, crop_right, crop_bottom))
                base.paste(trojan_cropped, (x0, y0), trojan_cropped)
            # else: completely out of view; return unchanged

        # Convert back to sample's original channel count
        if sc == 1:
            return np.asarray(base.convert("L"), dtype=np.uint8)
        elif sc == 3:
            return np.asarray(base.convert("RGB"), dtype=np.uint8)
        else:  # sc == 4
            return np.asarray(base.convert("RGBA"), dtype=np.uint8)
