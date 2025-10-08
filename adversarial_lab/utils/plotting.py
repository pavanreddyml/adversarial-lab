import io
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adversarial_lab.db import DB

from typing import Any, List, Literal, Optional, Dict, Tuple, Union


class Plotting:
    def __init__(self, db: DB, table_name: str, plot_config: Optional[dict] = None):
        self.db = db
        self.table_name = table_name

        plot_config = plot_config or {}

        self.plot_config = {
            "height": plot_config.get("height", 500),
            "width": plot_config.get("width", 800),
            "template": plot_config.get("template", "plotly_dark"),
            "show_title": plot_config.get("show_title", True),
            "show_legend": plot_config.get("show_legend", True),
            # image grid controls
            "grid_rows": plot_config.get("grid_rows"),  # optional; auto if None
            "grid_cols": plot_config.get("grid_cols", 3),  # default to 3 columns
        }

    @classmethod
    def valid_noise_stats(cls) -> List[str]:
        return [
            "mean", "median", "std", "min", "max",
            "var", "p25", "p75", "p_custom_x", "iqr", "skew", "kurtosis"
        ]

    def _get_column(self, column: str) -> pd.DataFrame:
        query = f"SELECT epoch_num, {column} FROM {self.table_name};"
        try:
            rows = self.db.execute_query(query)
            if rows is None:
                raise ValueError(
                    f"No data returned for column '{column}' from the database.")
            df = pd.DataFrame(rows)
        except Exception as e:
            raise ValueError(
                f"Column '{column}' does not exist in the database.") from e

        try:
            df[column] = df[column].apply(json.loads)
        except Exception:
            pass
        return df

    def _axis_color(self) -> Tuple[str, str]:
        background_color = "black" if self.plot_config.get(
            "template", "") == "plotly_dark" else "white"
        title_color = "white" if background_color == "black" else "black"
        return background_color, title_color
    
    def _try_get_value_for_epoch(self, column: str, epoch_num: int) -> Optional[Any]:
        """Return single value for a column at a given epoch, or None if missing/invalid."""
        try:
            rows = self.db.execute_query(
                f"SELECT {column} FROM {self.table_name} WHERE epoch_num = ? LIMIT 1",
                (epoch_num,),
            )
            if rows and column in rows[0]:
                return rows[0][column]
        except Exception:
            return None
        return None

    def _infer_classes_and_confidences(
        self,
        zero_epoch: int,
        target_epoch: int
    ) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
        # Original (prefer epoch 0 metadata)
        orig_idx = None
        for col in ["original_class_idx", "true_label", "label", "class_idx"]:
            val = self._try_get_value_for_epoch(col, zero_epoch)
            if val is not None:
                try:
                    orig_idx = int(val)
                    break
                except Exception:
                    pass

        orig_conf = None
        for col in ["original_confidence", "true_confidence", "orig_confidence"]:
            val = self._try_get_value_for_epoch(col, zero_epoch)
            if val is not None:
                try:
                    orig_conf = float(val)
                    break
                except Exception:
                    pass

        # Predicted (prefer explicit columns at target epoch)
        pred_idx = None
        for col in ["predicted_class_idx", "pred_class_idx"]:
            val = self._try_get_value_for_epoch(col, target_epoch)
            if val is not None:
                try:
                    pred_idx = int(val)
                    break
                except Exception:
                    pass

        pred_conf = None
        for col in ["predicted_confidence", "pred_confidence"]:
            val = self._try_get_value_for_epoch(col, target_epoch)
            if val is not None:
                try:
                    pred_conf = float(val)
                    break
                except Exception:
                    pass

        # Fallback: derive predicted from epoch_predictions at target epoch
        if pred_idx is None or pred_conf is None:
            try:
                df_preds = self._get_column("epoch_predictions")
                row = df_preds[df_preds["epoch_num"] == target_epoch]
                if not row.empty:
                    preds_dict = row.iloc[0]["epoch_predictions"]
                    if isinstance(preds_dict, dict) and preds_dict:
                        # keys may be strings; convert to (int_key, float_val)
                        items = []
                        for k, v in preds_dict.items():
                            try:
                                items.append((int(k), float(v)))
                            except Exception:
                                continue
                        if items:
                            items.sort(key=lambda x: -x[1])
                            top_k, top_v = items[0]
                            if pred_idx is None:
                                pred_idx = top_k
                            if pred_conf is None:
                                pred_conf = top_v
            except Exception:
                pass

        return orig_idx, orig_conf, pred_idx, pred_conf

    # -----------------------------
    # Plots
    # -----------------------------
    def plot_losses(
        self,
        plot_total_loss: bool = True,
        plot_main_loss: bool = True,
        plot_penalties: bool = True,
        penalty_idx: Optional[List[int]] = None,
        config: Optional[dict] = None,
        fixed_axes: bool = True,
        yrange: Optional[Tuple[float, float]] = None,
    ):
        config = config or {}

        df = self._get_column("epoch_losses")
        fig = go.Figure()
        keys_seen = {}

        y_min, y_max = float("inf"), float("-inf")

        for _, row in df.iterrows():
            epoch = row["epoch_num"]

            if not isinstance(row["epoch_losses"], dict):
                continue

            for key, val in row["epoch_losses"].items():
                if not (
                    (plot_total_loss and key == "Total Loss") or
                    (plot_main_loss and key.startswith("Loss ")) or
                    (plot_penalties and key.startswith("Penalty "))
                ):
                    continue

                if penalty_idx is not None and key.startswith("Penalty "):
                    try:
                        idx = int(key.split("Penalty ")[1].split(":")[0])
                        if idx not in penalty_idx:
                            continue
                    except (ValueError, IndexError):
                        continue

                if key not in keys_seen:
                    keys_seen[key] = {"x": [], "y": []}
                keys_seen[key]["x"].append(epoch)
                keys_seen[key]["y"].append(val)

                if isinstance(val, (int, float)):
                    y_min = min(y_min, float(val))
                    y_max = max(y_max, float(val))

        if penalty_idx is not None:
            if not any(k.startswith("Penalty ") and any(str(p) in k for p in penalty_idx) for k in keys_seen):
                warnings.warn(f"Penalty index {penalty_idx} not found in any epoch.")

        for key, vals in keys_seen.items():
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=key))

        if yrange is not None:
            yaxis_range = list(yrange)
        elif fixed_axes and y_min < y_max:
            pad = 0.05 * (y_max - y_min)
            yaxis_range = [y_min - pad, y_max + pad]
        else:
            yaxis_range = None

        fig.update_layout(
            title=config.get("title", "Losses Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            yaxis_title=config.get("yaxis_title", "Loss Value"),
            template=self.plot_config["template"],
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"],
            yaxis=dict(range=yaxis_range) if yaxis_range is not None else {}
        )
        fig.show()

    def plot_noise_statistics(
        self,
        stats: Union[str, List[str]],
        config: Optional[dict] = None,
        fixed_axes: bool = True,
        yrange: Optional[Tuple[float, float]] = None,
    ):
        if isinstance(stats, str):
            stats = [stats]
        if not stats:
            raise ValueError("`stats` is required and must contain at least one stat.")

        allowed = set(self.valid_noise_stats())
        invalid = [s for s in stats if s not in allowed]
        if invalid:
            warnings.warn(f"Ignoring invalid stats: {invalid}")
            stats = [s for s in stats if s in allowed]
        if not stats:
            raise ValueError("No valid stats to plot after validation.")

        config = config or {}
        df = self._get_column("epoch_noise_stats")
        if df.empty:
            return

        series: Dict[str, Dict[str, List]] = {}
        y_min, y_max = float("inf"), float("-inf")

        for _, row in df.iterrows():
            epoch = row["epoch_num"]
            stats_dict = row["epoch_noise_stats"]
            if not isinstance(stats_dict, dict):
                continue

            for stat in stats:
                if stat not in stats_dict:
                    continue
                if stat not in series:
                    series[stat] = {"x": [], "y": []}
                val = stats_dict[stat]
                series[stat]["x"].append(epoch)
                series[stat]["y"].append(val)
                try:
                    y_min = min(y_min, float(val))
                    y_max = max(y_max, float(val))
                except Exception:
                    pass

        fig = go.Figure()
        for stat, vals in series.items():
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=stat)
            )

        if yrange is not None:
            rng = list(yrange)
        elif fixed_axes and y_min < y_max:
            pad = 0.05 * (y_max - y_min)
            rng = [y_min - pad, y_max + pad]
        else:
            rng = None

        fig.update_layout(
            title=config.get("title", "Noise Statistics Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            yaxis_title=config.get("yaxis_title", "Value"),
            template=self.plot_config["template"],
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"],
            yaxis=dict(range=rng) if rng is not None else {}
        )
        fig.show()

    def plot_predictions(
        self,
        n: int = 10,
        idx: Optional[List[int]] = None,
        config: Optional[dict] = None,
        fixed_axes: bool = True,
        yrange: Optional[Tuple[float, float]] = None,
    ) -> None:
        if n > 20:
            warnings.warn(
                "Requested top-k predictions exceeds maximum of 20. Defaulting to 20.")
            n = 20

        if idx is not None:
            if len(idx) > 20:
                warnings.warn(
                    "Provided index list exceeds maximum of 20. Using first 20 entries.")
                idx = idx[:20]
            idx = [str(i) for i in idx]

        config = config or {}
        df = self._get_column("epoch_predictions")
        fig = go.Figure()

        keys_seen: Dict[str, Dict[str, List]] = {}
        y_min, y_max = float("inf"), float("-inf")

        for _, row in df.iterrows():
            epoch = row["epoch_num"]
            predictions: Dict[str, float] = row["epoch_predictions"]
            if not isinstance(predictions, dict):
                continue

            if idx is not None:
                selected = [(k, predictions[k])
                            for k in idx if k in predictions]
            else:
                selected = sorted(predictions.items(), key=lambda x: -x[1])[:n]

            for pred_idx, val in selected:
                if pred_idx not in keys_seen:
                    keys_seen[pred_idx] = {"x": [], "y": []}
                keys_seen[pred_idx]["x"].append(epoch)
                keys_seen[pred_idx]["y"].append(val)
                try:
                    y_min = min(y_min, float(val))
                    y_max = max(y_max, float(val))
                except Exception:
                    pass

        for pred_idx in sorted(keys_seen, key=lambda x: int(x)):
            vals = keys_seen[pred_idx]
            legend_name = f"Idx {pred_idx}"
            fig.add_trace(go.Scatter(
                x=vals["x"], y=vals["y"], mode="lines+markers", name=legend_name))

        if yrange is not None:
            yaxis_range = list(yrange)
        elif fixed_axes and y_min < y_max:
            pad = 0.05 * (y_max - y_min)
            yaxis_range = [y_min - pad, y_max + pad]
        else:
            yaxis_range = None

        fig.update_layout(
            title=config.get("title", "Top Predictions Over Epochs") if self.plot_config["show_title"] else None,
            xaxis_title=config.get("xaxis_title", "Epoch"),
            yaxis_title=config.get("yaxis_title", "Prediction Value"),
            template=self.plot_config["template"],
            height=self.plot_config["height"],
            width=self.plot_config["width"],
            showlegend=self.plot_config["show_legend"],
            yaxis=dict(range=yaxis_range) if yaxis_range is not None else {}
        )
        fig.show()

    def plot_samples_db(
        self,
        include: Optional[List[Literal[
            "original_image", "preprocessed_image",
            "noise", "normalized_noise", "noised_sample"
        ]]] = None,
        epoch: Optional[int] = None,
        class_dict: Optional[Dict[int, str]] = None,
    ):
        include = include or ["original_image", "preprocessed_image",
                              "noise", "normalized_noise", "noised_sample"]
        computed_columns = {"normalized_noise", "noised_sample"}
        required_columns = (set(include) - computed_columns) | {"noise"}

        epochs = self.db.execute_query(
            f"SELECT epoch_num FROM {self.table_name} ORDER BY epoch_num ASC")
        if not epochs:
            raise RuntimeError("No epoch data found.")

        all_epochs = [e["epoch_num"] for e in epochs]
        epoch = epoch if epoch is not None else (
            all_epochs[-2] if len(all_epochs) >= 2 else all_epochs[0])
        target_epoch = (0, epoch)

        cols = ['epoch_num'] + list(required_columns)
        placeholders = ','.join(['?'] * len(target_epoch))
        query = f"SELECT {', '.join(cols)} FROM {self.table_name} WHERE epoch_num IN ({placeholders})"
        row_data = self.db.execute_query(query, target_epoch)
        if not row_data:
            raise ValueError(f"No data found for epoch {target_epoch}")
        zero_row = [row for row in row_data if row["epoch_num"] == 0][0]
        epoch_row = [row for row in row_data if row["epoch_num"] == epoch][0]

        blobs = {}
        for col in required_columns:
            if col in {"original_image", "preprocessed_image"}:
                blobs[col] = zero_row.get(col)
            elif col in {"noise"}:
                blobs[col] = epoch_row.get(col)

        decoded = {}
        for col, blob in blobs.items():
            if blob is None:
                decoded[col] = None
                continue
            try:
                if col in ["original_image"]:
                    decoded[col] = np.array(Image.open(
                        io.BytesIO(blob)).convert("RGB"))
                elif col in ["preprocessed_image", "noise"]:
                    decoded[col] = pickle.loads(blob)
            except Exception as e:
                warnings.warn(f"Failed to decode {col}: {e}")
                decoded[col] = None

        # Infer classes/confidences
        orig_idx, orig_conf, pred_idx, pred_conf = self._infer_classes_and_confidences(0, epoch)

        def _name_for(idx: Optional[int]) -> Optional[str]:
            if idx is None or class_dict is None:
                return None
            try:
                return class_dict.get(int(idx))
            except Exception:
                return None

        orig_name = _name_for(orig_idx)
        pred_name = _name_for(pred_idx)

        images = []
        for col in include:
            noise = decoded.get("noise")
            orig = decoded.get("original_image")
            prep = decoded.get("preprocessed_image")
            try:
                if col == "preprocessed_image":
                    if prep is not None:
                        arr = prep.astype("float32")
                        min_val, max_val = arr.min(), arr.max()
                        if max_val - min_val < 1e-8:
                            arr[:] = 0
                        else:
                            arr = 2 * (arr - min_val) / (max_val - min_val) - 1
                        arr = ((arr + 1) / 2) * 255
                        arr = arr.astype("uint8")
                        img = Image.fromarray(arr)
                    else:
                        img = None
                elif col == "noise":
                    arr = decoded["noise"]
                    if arr is not None:
                        arr = arr.astype("float32")
                        base = np.full_like(arr, 127.0)
                        arr = base + arr
                        arr = np.clip(arr, 0, 255)
                        img = Image.fromarray(arr.astype("uint8"))
                    else:
                        img = None
                elif col == "normalized_noise":
                    arr = decoded["noise"]
                    if arr is not None:
                        arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
                        img = Image.fromarray((arr * 255).astype("uint8"))
                    else:
                        img = None
                elif col == "noised_sample":
                    base = None
                    if noise is not None:
                        if orig is not None and orig.shape == noise.shape:
                            base = orig
                        elif prep is not None and prep.shape == noise.shape:
                            base = prep
                    if base is not None:
                        arr = np.clip(base.astype(
                            "float32") + noise.astype("float32"), 0, 255).astype("uint8")
                        img = Image.fromarray(arr)
                    else:
                        img = None
                else:
                    arr = decoded.get(col)
                    if arr is None:
                        img = None
                    elif arr.ndim == 2:
                        img = Image.fromarray(arr)
                    elif arr.ndim == 3 and arr.shape[-1] in [1, 3]:
                        img = Image.fromarray(arr.astype("uint8"))
                    else:
                        raise ValueError(
                            f"Unsupported shape for image array: {arr.shape}")
            except Exception as e:
                warnings.warn(f"Failed to process {col} at epoch {epoch}: {e}")
                img = None
            images.append(img)

        background_color, title_color = self._axis_color()

        n_images = len(images)
        grid_cols = max(1, int(self.plot_config.get("grid_cols", 3)))
        grid_rows_cfg = self.plot_config.get("grid_rows")
        if grid_rows_cfg is None:
            rows = (n_images + grid_cols - 1) // grid_cols
        else:
            rows = max(1, int(grid_rows_cfg))
        cols = grid_cols
        aspect_ratio = (4, 4)

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * aspect_ratio[0], rows * aspect_ratio[1]),
                                 facecolor=background_color)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        def _fmt_class(name: Optional[str], idx: Optional[int], conf: Optional[float]) -> str:
            if idx is None and name is None:
                return ""
            parts = []
            if name is not None:
                parts.append(name)
            if idx is not None:
                parts.append(f"id:{int(idx)}")
            if conf is not None:
                try:
                    parts.append(f"conf:{float(conf):.4f}")
                except Exception:
                    pass
            return " (" + ", ".join(parts) + ")" if parts else ""

        for ax, img, key in zip(axes, images, include):
            ax.axis("off")
            if img is not None:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)

            if key == "original_image":
                suffix = _fmt_class(orig_name, orig_idx, orig_conf)
                tt = f"Original{suffix}"
            elif key == "normalized_noise":
                tt = "Normalized Noise"
            elif key == "noise":
                tt = "Noise"
            elif key == "preprocessed_image":
                suffix = _fmt_class(orig_name, orig_idx, orig_conf)
                tt = f"Preprocessed original{suffix}"
            elif key == "noised_sample":
                suffix = _fmt_class(pred_name, pred_idx, pred_conf)
                tt = f"Noised{suffix}"
            else:
                tt = str(key)
            ax.set_title(tt, color=title_color)

        for ax in axes[len(images):]:
            ax.axis("off")

        plt.tight_layout(pad=0.1)
        plt.show()

    @staticmethod
    def plot_samples(
        image: np.ndarray,
        noise: np.ndarray,
        include: Optional[List[Literal[
            "image", "noise", "normalized_noise", "noised_image"
        ]]] = None,
        config: Optional[dict] = None,
        class_dict: Optional[Dict[int, str]] = None,
        original_class_idx: Optional[int] = None,
        predicted_class_idx: Optional[int] = None,
        original_confidence: Optional[float] = None,
        predicted_confidence: Optional[float] = None,
    ) -> None:
        # config-driven grid; defaults: 2 rows x 2 cols
        config = config or {}
        include = include or ["image", "noise", "normalized_noise", "noised_image"]

        if image.shape[0] == 1:
            image = image[0]
        if noise.shape[0] == 1:
            noise = noise[0]

        def normalize_for_display(img):
            if np.issubdtype(img.dtype, np.integer):
                return img / 255.0
            elif np.issubdtype(img.dtype, np.floating):
                return (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        plots: List[np.ndarray] = []
        titles: List[str] = []

        def _fmt_class(name: Optional[str], idx: Optional[int], conf: Optional[float]) -> str:
            if idx is None and name is None:
                return ""
            parts = []
            if name is not None:
                parts.append(name)
            if idx is not None:
                parts.append(f"id:{int(idx)}")
            if conf is not None:
                parts.append(f"conf:{float(conf):.4f}")
            return " (" + ", ".join(parts) + ")"

        orig_name = class_dict.get(int(original_class_idx), None) if (class_dict is not None and original_class_idx is not None) else None
        pred_name = class_dict.get(int(predicted_class_idx), None) if (class_dict is not None and predicted_class_idx is not None) else None

        if "image" in include:
            image_disp = normalize_for_display(image)
            plots.append(image_disp)
            titles.append(f"Original{_fmt_class(orig_name, original_class_idx, original_confidence)}")

        if "noised_image" in include:
            noisy_image = image + noise
            noisy_image_disp = normalize_for_display(noisy_image)
            plots.append(noisy_image_disp)
            titles.append(f"Noised{_fmt_class(pred_name, predicted_class_idx, predicted_confidence)}")

        if "noise" in include:
            base = np.full_like(noise, 127.0)
            noise_display = base + noise
            plots.append(noise_display.astype("uint8"))
            titles.append("Noise")

        if "normalized_noise" in include:
            norm_noise = (noise - np.min(noise)) / (np.ptp(noise) + 1e-8)
            plots.append(norm_noise)
            titles.append("Normalized Noise")

        n = len(plots)

        cols = int(config.get("grid_cols", 2))
        rows = int(config.get("grid_rows", 2))
        total_slots = rows * cols
        if n > total_slots:
            rows = int(np.ceil(n / cols))
            total_slots = rows * cols

        fig_w, fig_h = 5 * cols, 5 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i in range(total_slots):
            ax = axes_flat[i]
            if i < n:
                img = plots[i]
                title = titles[i]
                ax.imshow(img, cmap='gray' if (isinstance(img, np.ndarray) and img.ndim == 2) else None)
                if config.get("show_subtitle", True):
                    ax.set_title(title)
            ax.axis('off')

        if config.get("title") is not None:
            fig.suptitle(config["title"])

        plt.tight_layout()
        plt.show()
