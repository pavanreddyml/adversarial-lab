from PIL import Image
import numpy as np
import random
from typing import Any, List, Tuple

from adversarial_lab.core.noise_generators import NoiseGenerator


class TrojanImageNoiseGenerator(NoiseGenerator):
    def __init__(
        self,
        trojans: List[str | np.ndarray],
        size: Tuple[int, int] | int = (32, 32),
        position: Tuple[float, float] = (10.0, 10.0),
        rotation: Tuple[float, float] | float = (0.0, 0.0),
        keep_aspect_ratio: bool = True,
        fit_to_size: bool = True,
        coerce_out_of_bound: bool = True,
        alpha: float = 1.0,
    ):
        if len(trojans) == 0:
            raise ValueError("At least one trojan must be provided.")

        if all(isinstance(t, np.ndarray) for t in trojans):
            self.trojans = trojans
        elif all(isinstance(t, str) for t in trojans):
            if not all(t.lower().endswith((".png", ".jpg", ".jpeg")) for t in trojans):
                raise ValueError("Trojans supported formats are .png, .jpg and .jpeg")
            self.trojans = [np.array(Image.open(t)) for t in trojans]
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

        if isinstance(rotation, (int, float)):
            if not (-360.0 <= float(rotation) <= 360.0):
                raise ValueError("Rotation angle must be between -360 and 360 degrees.")
            self.rotation = (float(rotation), float(rotation))
        elif isinstance(rotation, tuple) and len(rotation) == 2:
            r0, r1 = float(rotation[0]), float(rotation[1])
            if not (-360.0 <= r0 <= 360.0 and -360.0 <= r1 <= 360.0):
                raise ValueError("Rotation angles must be between -360 and 360 degrees.")
            self.rotation = (r0, r1)
        else:
            raise ValueError("Rotation must be either a float angle or a tuple (min_deg, max_deg).")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0.")
        self.alpha = float(alpha)
        self._current_alpha = None

        self.keep_aspect_ratio = keep_aspect_ratio
        self.fit_to_size = fit_to_size
        self.coerce_out_of_bound = coerce_out_of_bound

        self._RESIZE = getattr(Image, "LANCZOS", Image.BICUBIC)
        self._ROTATE = Image.BICUBIC

        self._last_mask_np = None

    def apply_noise(self, sample: np.ndarray, trojan_id: int, *args, **kwargs) -> Any:
        self._current_alpha = float(kwargs.get("alpha", self.alpha))
        raw_trojan = self.trojans[trojan_id]
        trojan = self._get_trojan_reshaped(sample, raw_trojan)[0]
        out = self._composite_hard_overwrite(sample, trojan, self.position, self.coerce_out_of_bound)
        self._current_alpha = None
        self._last_mask_np = None
        return out

    def get_num_trojans(self) -> int:
        return len(self.trojans)

    def _get_trojan_reshaped(self, sample: np.ndarray, trojan: np.ndarray) -> List[np.ndarray]:
        sample_h, sample_w, sample_c = self._shape_info(sample)
        target_w, target_h = self.size

        trojan_img = self._np_to_pil(trojan)
        trojan_img = self._convert_mode_to_match_sample(trojan_img, 1 if sample_c == 1 else 3)

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
        angle_deg = random.uniform(min_r, max_r)

        if resized.mode == "L":
            fill = 0
        else:
            fill = self._estimate_border_fillcolor(resized)

        try:
            rotated = resized.rotate(angle=angle_deg, resample=self._ROTATE, expand=True, fillcolor=fill)
        except TypeError:
            rotated = resized.rotate(angle=angle_deg, resample=self._ROTATE, expand=True)

        ones = Image.new("L", resized.size, 255)
        try:
            mask_im = ones.rotate(angle=angle_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
        except TypeError:
            mask_im = ones.rotate(angle=angle_deg, resample=Image.NEAREST, expand=True)
        mask_np = (np.asarray(mask_im, dtype=np.uint8) > 0)

        trojan_np = self._pil_to_np_matching_sample(rotated, 1 if sample_c == 1 else 3)
        self._last_mask_np = mask_np
        return [trojan_np]

    def _composite_hard_overwrite(
        self,
        sample_np: np.ndarray,
        trojan_np: np.ndarray,
        pos_percent: Tuple[float, float],
        coerce: bool,
    ) -> np.ndarray:
        sh, sw, sc = self._shape_info(sample_np)

        if sc == 1:
            base = Image.fromarray(sample_np.astype(np.uint8), mode="L")
            canvas = np.zeros((sh, sw), dtype=np.uint8)
            mask_canvas = np.zeros((sh, sw), dtype=bool)
        elif sc == 3:
            base = Image.fromarray(sample_np.astype(np.uint8), mode="RGB")
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            mask_canvas = np.zeros((sh, sw), dtype=bool)
        elif sc == 4:
            base = Image.fromarray(sample_np[:, :, :3].astype(np.uint8), mode="RGB")
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            mask_canvas = np.zeros((sh, sw), dtype=bool)
        else:
            raise ValueError("sample must have 1, 3, or 4 channels.")

        trojan_img = self._np_to_pil(trojan_np)
        trojan_img = self._convert_mode_to_match_sample(trojan_img, 1 if base.mode == "L" else 3)

        fw, fh = trojan_img.size
        mask_np = self._last_mask_np
        if mask_np is None:
            if base.mode == "L":
                mask_np = np.ones((fh, fw), dtype=bool)
            else:
                mask_np = np.ones((fh, fw), dtype=bool)

        px_pct, py_pct = pos_percent
        center_x = int(round((px_pct / 100.0) * sw))
        center_y_from_bottom = int(round((py_pct / 100.0) * sh))
        center_y = sh - center_y_from_bottom

        x = center_x - fw // 2
        y = center_y - fh // 2

        if coerce:
            x = max(0, min(x, sw - fw))
            y = max(0, min(y, sh - fh))

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(sw, x + fw)
        y1 = min(sh, y + fh)

        if x0 < x1 and y0 < y1:
            crop_left = x0 - x
            crop_top = y0 - y
            crop_right = crop_left + (x1 - x0)
            crop_bottom = crop_top + (y1 - y0)

            trojan_cropped = trojan_img.crop((crop_left, crop_top, crop_right, crop_bottom))
            trojan_arr = np.asarray(trojan_cropped, dtype=np.uint8)

            mask_crop = mask_np[crop_top:crop_bottom, crop_left:crop_right]

            if base.mode == "L":
                canvas[y0:y1, x0:x1] = trojan_arr
                mask_canvas[y0:y1, x0:x1] = mask_crop
            else:
                canvas[y0:y1, x0:x1, :] = trojan_arr
                mask_canvas[y0:y1, x0:x1] = mask_crop

        a = self._current_alpha if self._current_alpha is not None else self.alpha
        a = float(min(1.0, max(0.0, a)))

        if sc == 1:
            base_np = np.array(base, dtype=np.uint8, copy=True)
            m = mask_canvas
            if m.any():
                b = base_np[m].astype(np.float32)
                t = canvas[m].astype(np.float32)
                base_np[m] = np.clip((1.0 - a) * b + a * t, 0, 255).astype(np.uint8)
            return base_np
        else:
            base_np = np.array(base, dtype=np.uint8, copy=True)
            m = mask_canvas
            if m.any():
                b = base_np[m].astype(np.float32)
                t = canvas[m].astype(np.float32)
                base_np[m] = np.clip((1.0 - a) * b + a * t, 0, 255).astype(np.uint8)
            if sc == 3:
                return base_np
            else:
                a_ch = np.full((sh, sw, 1), 255, dtype=np.uint8)
                return np.concatenate([base_np, a_ch], axis=2)

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
    def _convert_mode_to_match_sample(img: Image.Image, sample_c: int) -> Image.Image:
        if sample_c == 1:
            return img.convert("L") if img.mode != "L" else img
        else:
            return img.convert("RGB") if img.mode != "RGB" else img

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
