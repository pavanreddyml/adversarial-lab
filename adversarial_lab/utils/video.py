import io
import json
import pickle
import imageio
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm  # NEW: progress bars

from adversarial_lab.db import DB

from typing import List, Optional, Literal, Tuple, Dict, Any


class VideoPlotter:
    def __init__(
        self,
        db: DB,
        table_name: str,
        plot_config: Optional[dict] = None,
        class_dict: Optional[Dict[int, str]] = None,
        original_class_idx: Optional[int] = None,
    ):
        self.db = db
        self.table_name = table_name
        plot_config = plot_config or {}

        self.plot_config = {
            "height": plot_config.get("height", 500),
            "width": plot_config.get("width", 800),
            "template": plot_config.get("template", "plotly_dark"),
            "show_title": plot_config.get("show_title", True),
            "show_legend": plot_config.get("show_legend", True),
            "grid_rows": plot_config.get("grid_rows"),  # image grid rows (optional; auto if None)
            "grid_cols": plot_config.get("grid_cols", 2),  # image grid cols default
        }
        self.class_dict = class_dict or {}
        self.original_class_idx = original_class_idx
        self._cache: Dict[int, Dict[str, Any]] = {}  # populated on first make_video call

    def _axis_colors(self) -> Tuple[str, str]:
        bg = "black" if self.plot_config.get("template", "") == "plotly_dark" else "white"
        fg = "white" if bg == "black" else "black"
        return bg, fg

    def _load_cache_once(self) -> None:
        """Load all rows/columns once; decode JSON fields; keep blobs as-is (decode later)."""
        if self._cache:
            return
        rows = self.db.execute_query(f"SELECT * FROM {self.table_name}")
        if not rows:
            raise RuntimeError("No data found in table.")
        for r in rows:
            ep = r.get("epoch_num")
            if ep is None:
                continue

            def _maybe_json(v):
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except Exception:
                        return v
                return v

            item = {k: _maybe_json(v) for k, v in r.items()}
            self._cache[int(ep)] = item

    def _epochs_sorted(self) -> List[int]:
        eps = sorted(self._cache.keys())
        if not eps:
            raise RuntimeError("Cache empty.")
        return eps

    def _get_epoch_row(self, epoch: int) -> Dict[str, Any]:
        return self._cache.get(epoch, {})

    def _to_uint8_for_display(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[2] == 4:  # drop alpha if present
            a = a[..., :3]
        a = a.astype(np.float32)

        amin = float(np.min(a))
        amax = float(np.max(a))
        ptp = amax - amin

        if not np.isfinite(amin) or not np.isfinite(amax):
            return np.zeros_like(a, dtype=np.uint8)

        if ptp < 1e-12:
            # Near-constant: render as mid-gray
            return np.full_like(a, 127, dtype=np.uint8)

        a = (a - amin) / (ptp + 1e-12)
        a = (a * 255.0).clip(0, 255)
        return a.astype(np.uint8)

    def _decode_images_for_epoch(
        self,
        epoch: int,
        include_images: List[str]
    ) -> Dict[str, Optional[np.ndarray]]:
        out: Dict[str, Optional[np.ndarray]] = {k: None for k in include_images}
        zero = self._get_epoch_row(0)
        cur = self._get_epoch_row(epoch)

        orig_blob = zero.get("original_image")
        prep_blob = zero.get("preprocessed_image")
        noise_pickled = cur.get("noise")

        orig_disp = None
        if "original_image" in include_images and orig_blob is not None:
            try:
                orig_disp = np.array(Image.open(io.BytesIO(orig_blob)).convert("RGB"))
                out["original_image"] = orig_disp
            except Exception as e:
                warnings.warn(f"Decode original_image failed: {e}")

        prep_raw = None
        prep_disp = None
        if "preprocessed_image" in include_images or "noised_sample" in include_images:
            if prep_blob is not None:
                try:
                    prep_raw = pickle.loads(prep_blob)
                    prep_disp = self._to_uint8_for_display(prep_raw)
                    if "preprocessed_image" in include_images:
                        out["preprocessed_image"] = prep_disp
                except Exception as e:
                    warnings.warn(f"Decode preprocessed_image failed: {e}")

        noise = None
        if ("noise" in include_images or "normalized_noise" in include_images or "noised_sample" in include_images) and noise_pickled is not None:
            try:
                noise = pickle.loads(noise_pickled)
                if "noise" in include_images:
                    arr = np.asarray(noise, dtype=np.float32)
                    base = np.full_like(arr, 127.0)
                    arr = np.clip(base + arr, 0, 255)
                    out["noise"] = arr.astype("uint8")
            except Exception as e:
                warnings.warn(f"Decode noise failed: {e}")

        if "normalized_noise" in include_images and noise is not None:
            try:
                out["normalized_noise"] = self._to_uint8_for_display(np.asarray(noise))
            except Exception as e:
                warnings.warn(f"Compute normalized_noise failed: {e}")

        if "noised_sample" in include_images and noise is not None:
            try:
                n = np.asarray(noise, dtype=np.float32)
                base = None
                if orig_disp is not None and orig_disp.shape == n.shape:
                    base = orig_disp.astype(np.float32)
                elif prep_disp is not None and prep_disp.shape == n.shape:
                    base = prep_disp.astype(np.float32)

                if base is not None:
                    arr = np.clip(base + n, 0, 255).astype("uint8")
                    out["noised_sample"] = arr
            except Exception as e:
                warnings.warn(f"Compute noised_sample failed: {e}")

        return out

    def _infer_original_idx(self) -> Optional[int]:
        if self.original_class_idx is not None:
            return self.original_class_idx
        zero = self._get_epoch_row(0)
        for col in ["original_class_idx", "true_label", "label", "class_idx"]:
            if zero.get(col) is not None:
                try:
                    return int(zero[col])
                except Exception:
                    continue
        return None

    def _infer_confidences(self, epoch: int) -> Tuple[Optional[float], Optional[float]]:
        zero = self._get_epoch_row(0)
        cur = self._get_epoch_row(epoch)

        orig_conf = None
        for col in ["original_confidence", "true_confidence", "orig_confidence"]:
            if zero.get(col) is not None:
                try:
                    orig_conf = float(zero[col])
                    break
                except Exception:
                    pass

        pred_conf = None
        for col in ["predicted_confidence", "pred_confidence"]:
            if cur.get(col) is not None:
                try:
                    pred_conf = float(cur[col])
                    break
                except Exception:
                    pass

        if pred_conf is None:
            preds = cur.get("epoch_predictions")
            if isinstance(preds, dict) and preds:
                try:
                    pred_conf = float(max(map(float, preds.values())))
                except Exception:
                    pass

        return orig_conf, pred_conf

    def _infer_pred_idx_for_epoch(self, epoch: int) -> Optional[int]:
        preds = self._get_epoch_row(epoch).get("epoch_predictions")
        if isinstance(preds, dict) and preds:
            try:
                best_k = max(preds.items(), key=lambda kv: float(kv[1]))[0]
                return int(best_k)
            except Exception:
                return None
        return None

    def _fmt_class_triplet(self, name: Optional[str], idx: Optional[int], conf: Optional[float]) -> str:
        parts = []
        if name:
            parts.append(name)
        if idx is not None:
            parts.append(f"id={int(idx)}")
        if conf is not None:
            try:
                parts.append(f"conf={float(conf):.4f}")
            except Exception:
                pass
        return f" ({', '.join(parts)})" if parts else ""

    def _panel_title(self, key: str, epoch: int) -> str:
        cd = self.class_dict or {}
        orig_idx = self._infer_original_idx()
        orig_nm = cd.get(int(orig_idx)) if orig_idx is not None else None
        pred_idx = self._infer_pred_idx_for_epoch(epoch)
        pred_nm = cd.get(int(pred_idx)) if pred_idx is not None else None
        orig_conf, pred_conf = self._infer_confidences(epoch)

        if key == "original_image":
            return f"Original{self._fmt_class_triplet(orig_nm, orig_idx, orig_conf)}"
        if key == "preprocessed_image":
            return f"Preprocessed original{self._fmt_class_triplet(orig_nm, orig_idx, orig_conf)}"
        if key == "noised_sample":
            return f"Noised{self._fmt_class_triplet(pred_nm, pred_idx, pred_conf)}"
        if key == "normalized_noise":
            return "Normalized Noise"
        if key == "noise":
            return "Noise"
        return key

    def _legend_label(self, idx: str) -> str:
        try:
            name = (self.class_dict or {}).get(int(idx))
            return f"{idx} ({name})" if name else f"Idx {idx}"
        except Exception:
            return f"Idx {idx}"

    def _title_suffix_pair(self, epoch: int) -> str:
        cd = self.class_dict or {}
        orig_idx = self._infer_original_idx()
        pred_idx = self._infer_pred_idx_for_epoch(epoch)
        parts = []
        if orig_idx is not None:
            nm = cd.get(int(orig_idx))
            parts.append(f"orig: {orig_idx}" + (f" ({nm})" if nm else ""))
        if pred_idx is not None:
            nm = cd.get(int(pred_idx))
            parts.append(f"pred: {pred_idx}" + (f" ({nm})" if nm else ""))
        return " | " + " · ".join(parts) if parts else ""

    def _fig_to_rgb_array(self, fig) -> np.ndarray:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        if hasattr(fig.canvas, "buffer_rgba"):
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            return buf[..., :3].copy()
        renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
        if renderer is not None and hasattr(renderer, "buffer_rgba"):
            buf = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            return buf[..., :3].copy()
        if hasattr(fig.canvas, "tostring_argb"):
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            rgba = argb[:, :, [1, 2, 3, 0]]
            return rgba[..., :3].copy()
        raise RuntimeError("Could not extract RGBA buffer from Matplotlib canvas.")

    def make_video(
        self,
        include: Optional[List[Literal[
            "original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample",
            "losses", "predictions", "noise_stats"
        ]]] = None,
        fps: int = 1,
        save_path: str = "output_video.mp4",
        # Losses config (prefixed)
        losses_plot_total_loss: bool = True,
        losses_plot_main_loss: bool = True,
        losses_plot_penalties: bool = True,
        losses_penalty_idx: Optional[List[int]] = None,
        # Predictions config (prefixed)
        predictions_topk: int = 10,
        predictions_idx: Optional[List[int]] = None,
        # Noise stats config (prefixed)
        noise_stats_list: Optional[List[Literal[
            "mean", "median", "std", "min", "max", "var",
            "p25", "p75", "p_custom_x", "iqr", "skew", "kurtosis"
        ]]] = None,
    ):
        self._load_cache_once()

        include = include or [
            "original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample",
            "losses", "predictions", "noise_stats"
        ]

        if predictions_topk > 20:
            warnings.warn("predictions_topk exceeds 20; capping to 20.")
            predictions_topk = 20
        if predictions_idx is not None and len(predictions_idx) > 20:
            warnings.warn("predictions_idx length > 20; using first 20.")
            predictions_idx = predictions_idx[:20]

        # Epochs
        epochs = self._epochs_sorted()[:-1]
        if len(epochs) < 2:
            raise RuntimeError("At least two epochs required (including epoch 0).")

        losses_data: Dict[str, Dict[int, float]] = {}
        lx_min = float("inf"); lx_max = float("-inf")
        ly_min = float("inf"); ly_max = float("-inf")
        if "losses" in include:
            for ep in epochs:
                row = self._get_epoch_row(ep)
                losses = row.get("epoch_losses")
                if not isinstance(losses, dict):
                    continue
                lx_min = min(lx_min, ep); lx_max = max(lx_max, ep)
                for key, val in losses.items():
                    if not (
                        (losses_plot_total_loss and key == "Total Loss") or
                        (losses_plot_main_loss and key.startswith("Loss ")) or
                        (losses_plot_penalties and key.startswith("Penalty "))
                    ):
                        continue
                    if losses_penalty_idx is not None and key.startswith("Penalty "):
                        try:
                            idx = int(key.split("Penalty ")[1].split(":")[0])
                            if idx not in losses_penalty_idx:
                                continue
                        except Exception:
                            continue
                    try:
                        y = float(val)
                    except Exception:
                        continue
                    losses_data.setdefault(key, {})[ep] = y
                    ly_min = min(ly_min, y); ly_max = max(ly_max, y)
            if losses_penalty_idx is not None:
                if not any(k.startswith("Penalty ") and any(str(p) in k for p in losses_penalty_idx) for k in losses_data):
                    warnings.warn(f"Penalty indices {losses_penalty_idx} not found in losses.")

        preds_data: Dict[str, Dict[int, float]] = {}
        px_min = float("inf"); px_max = float("-inf")
        py_min = float("inf"); py_max = float("-inf")
        if "predictions" in include:
            for ep in epochs:
                row = self._get_epoch_row(ep)
                preds = row.get("epoch_predictions")
                if not isinstance(preds, dict):
                    continue
                px_min = min(px_min, ep); px_max = max(px_max, ep)
                for k, v in preds.items():
                    try:
                        y = float(v)
                    except Exception:
                        continue
                    preds_data.setdefault(str(k), {})[ep] = y
                    py_min = min(py_min, y); py_max = max(py_max, y)

        noise_data: Dict[str, Dict[int, float]] = {}
        nx_min = float("inf"); nx_max = float("-inf")
        noise_ranges: Dict[str, Tuple[float, float]] = {}
        if "noise_stats" in include:
            chosen = noise_stats_list or [
                "mean", "median", "std", "min", "max", "var", "p25", "p75",
                "p_custom_x", "iqr", "skew", "kurtosis"
            ]
            for ep in epochs:
                row = self._get_epoch_row(ep)
                st = row.get("epoch_noise_stats")
                if not isinstance(st, dict):
                    continue
                nx_min = min(nx_min, ep); nx_max = max(nx_max, ep)
                for s in chosen:
                    if s not in st:
                        continue
                    try:
                        y = float(st[s])
                    except Exception:
                        continue
                    noise_data.setdefault(s, {})[ep] = y
                    ymin, ymax = noise_ranges.get(s, (float("inf"), float("-inf")))
                    noise_ranges[s] = (min(ymin, y), max(ymax, y))

        bg, fg = self._axis_colors()
        frames: List[np.ndarray] = []

        image_keys = [k for k in include if k in {"original_image", "preprocessed_image", "noise", "normalized_noise", "noised_sample"}]
        want_losses = "losses" in include and bool(losses_data)
        want_preds = "predictions" in include and bool(preds_data)
        want_stats = "noise_stats" in include and bool(noise_data)

        img_cols = max(1, int(self.plot_config.get("grid_cols", 2)))
        grid_rows_cfg = self.plot_config.get("grid_rows")
        img_count = len(image_keys)
        img_rows = max(1, (img_count + img_cols - 1) // img_cols) if grid_rows_cfg is None else max(1, int(grid_rows_cfg))

        graph_rows = int(want_losses) + int(want_preds) + int(want_stats)
        total_rows = img_rows + graph_rows if graph_rows > 0 else img_rows

        fig_w = max(10, img_cols * 4)
        fig_h = max(6, img_rows * 4 + graph_rows * 3)

        for ep in tqdm([e for e in epochs if e != 0], desc="Rendering frames"):
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg)
            gs = GridSpec(total_rows, img_cols, figure=fig, hspace=0.35, wspace=0.15)

            decoded = self._decode_images_for_epoch(ep, image_keys) if image_keys else {}
            for idx_panel, key in enumerate(image_keys):
                r = idx_panel // img_cols
                c = idx_panel % img_cols
                ax = fig.add_subplot(gs[r, c])
                ax.set_facecolor(bg)
                ax.axis("off")
                arr = decoded.get(key)
                if arr is not None:
                    ax.imshow(arr, cmap="gray" if (isinstance(arr, np.ndarray) and arr.ndim == 2) else None, vmin=0, vmax=255 if isinstance(arr, np.ndarray) and arr.dtype == np.uint8 else None)
                ax.set_title(self._panel_title(key, ep), color=fg, fontsize=10)

            for k in range(len(image_keys), img_rows * img_cols):
                r = k // img_cols
                c = k % img_cols
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")
                ax.set_facecolor(bg)

            # Graphs
            current_row = img_rows
            tick_params = dict(colors=fg, labelcolor=fg)
            suffix_pair = self._title_suffix_pair(ep)

            # Losses
            if want_losses:
                ax = fig.add_subplot(gs[current_row, :])
                ax.set_facecolor(bg)
                if lx_min < lx_max:
                    ax.set_xlim(lx_min, lx_max)
                if ly_min < ly_max:
                    pad = 0.05 * (ly_max - ly_min)
                    ax.set_ylim(ly_min - pad, ly_max + pad)
                for key, series in losses_data.items():
                    xs = [x for x in sorted(series.keys()) if x <= ep]
                    if not xs:
                        continue
                    ys = [series[x] for x in xs]
                    ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.5, label=key)
                if self.plot_config["show_legend"]:
                    leg = ax.legend(ncol=2)
                    for t in leg.get_texts():
                        t.set_color(fg)
                    leg.get_frame().set_edgecolor(fg)
                    leg.get_frame().set_facecolor(bg)
                ax.set_title(f"Losses Over Epochs{suffix_pair}", color=fg)
                ax.tick_params(**tick_params)
                for spine in ax.spines.values():
                    spine.set_color(fg)
                ax.set_xlabel("Epoch", color=fg)
                ax.set_ylabel("Loss Value", color=fg)
                current_row += 1

            # Predictions (ONLY current epoch's top-k)
            if want_preds:
                ax = fig.add_subplot(gs[current_row, :])
                ax.set_facecolor(bg)

                cur_preds = self._get_epoch_row(ep).get("epoch_predictions")
                allowed_keys: List[str] = []
                if isinstance(cur_preds, dict) and cur_preds:
                    if predictions_idx is not None:
                        allowed_keys = [str(i) for i in predictions_idx if str(i) in cur_preds]
                    else:
                        allowed_keys = [str(k) for k, _ in sorted(cur_preds.items(), key=lambda x: float(x[1]), reverse=True)[:predictions_topk]]

                if px_min < px_max:
                    ax.set_xlim(px_min, px_max)
                if allowed_keys:
                    ymins, ymaxs = [], []
                    for k in allowed_keys:
                        ser = preds_data.get(k, {})
                        xs = [x for x in sorted(ser.keys()) if x <= ep]
                        if xs:
                            ys = [ser[x] for x in xs]
                            ymins.append(min(ys)); ymaxs.append(max(ys))
                    if ymins and ymaxs:
                        ymin = min(ymins); ymax = max(ymaxs)
                        if ymin < ymax:
                            pad = 0.05 * (ymax - ymin)
                            ax.set_ylim(ymin - pad, ymax + pad)

                for k in allowed_keys:
                    series = preds_data.get(k, {})
                    xs = [x for x in sorted(series.keys()) if x <= ep]
                    if not xs:
                        continue
                    ys = [series[x] for x in xs]
                    label = self._legend_label(k)
                    ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.5, label=label)

                if self.plot_config["show_legend"] and allowed_keys:
                    leg = ax.legend(ncol=2)
                    for t in leg.get_texts():
                        t.set_color(fg)
                    leg.get_frame().set_edgecolor(fg)
                    leg.get_frame().set_facecolor(bg)

                ax.set_title(f"Top Predictions Over Epochs (current top-{len(allowed_keys)}){suffix_pair}", color=fg)
                ax.tick_params(**tick_params)
                for spine in ax.spines.values():
                    spine.set_color(fg)
                ax.set_xlabel("Epoch", color=fg)
                ax.set_ylabel("Prediction Value", color=fg)
                current_row += 1

            # Noise stats
            if want_stats:
                ax = fig.add_subplot(gs[current_row, :])
                ax.set_facecolor(bg)
                if nx_min < nx_max:
                    ax.set_xlim(nx_min, nx_max)
                ymin = min((rng[0] for rng in noise_ranges.values() if rng[0] != float("inf")), default=None)
                ymax = max((rng[1] for rng in noise_ranges.values() if rng[1] != float("-inf")), default=None)
                if ymin is not None and ymax is not None and ymin < ymax:
                    pad = 0.05 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
                for stat, series in noise_data.items():
                    xs = [x for x in sorted(series.keys()) if x <= ep]
                    if not xs:
                        continue
                    ys = [series[x] for x in xs]
                    ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.5, label=stat)
                if self.plot_config["show_legend"]:
                    leg = ax.legend(ncol=2)
                    for t in leg.get_texts():
                        t.set_color(fg)
                    leg.get_frame().set_edgecolor(fg)
                    leg.get_frame().set_facecolor(bg)
                ax.set_title(f"Noise Statistics Over Epochs{suffix_pair}", color=fg)
                ax.tick_params(**tick_params)
                for spine in ax.spines.values():
                    spine.set_color(fg)
                ax.set_xlabel("Epoch", color=fg)
                ax.set_ylabel("Value", color=fg)

            # Capture frame
            frames.append(self._fig_to_rgb_array(fig))
            plt.close(fig)

        if not frames:
            raise RuntimeError("No frames were generated; check your data or 'include' list.")

        # Encode with progress bar
        writer = imageio.get_writer(save_path, fps=fps)
        try:
            for frame in tqdm(frames, desc="Encoding video"):
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"Video saved to {save_path}")
