from . import HandlerBase

import os
import random
from typing import Dict, List, Any, Tuple
import warnings

from PIL import Image
import numpy as np


class ImagesFromDirectory(HandlerBase):
    ALLOWED_EXTS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')

    def __init__(
        self,
        directory_path,
        output_path=None,
        batch_size=1,
        strategy='sequential',
        overwrite=False,
        include_alpha=False,
        *args,
        **kwargs
    ):
        super().__init__(batch_size=batch_size, *args, **kwargs)
        self.directory_path = directory_path

        if output_path is None:
            self.output_path = directory_path
            self.same_read_write = True
            warnings.warn(
                "Write path not provided. Using directory_path as write_path. "
                "To overwrite in same directory, set `overwrite=True`. Not doing so will raise an error."
            )
        else:
            if not os.path.isdir(output_path):
                raise ValueError(f"Output path {output_path} is not a valid directory.")
            self.output_path = output_path
            self.same_read_write = os.path.abspath(directory_path) == os.path.abspath(output_path)

        if strategy not in ('sequential', 'random'):
            raise ValueError(f"Unsupported strategy '{strategy}'. Use 'sequential' or 'random'.")
        self.strategy = strategy

        self.overwrite = overwrite
        self.include_alpha = include_alpha

        self.image_files: Dict[str, List[str]] = {"__root": []}
        self.initialize()

    def _choose_path(self, class_name: str, file_name: str, write=False) -> str:
        base_path = self.output_path if write else self.directory_path
        if class_name == "__root":
            return os.path.join(base_path, file_name)
        return os.path.join(base_path, class_name, file_name)

    def load(self, class_name: str = None):
        if class_name is not None and class_name not in self.image_files and class_name != "__root":
            raise ValueError(f"Class name {class_name} not found in directory.")

        if class_name is not None:
            files = self.image_files.get(class_name, [])
            if not files:
                return None
            if self.strategy == 'sequential':
                file_name = files.pop(0)
            else:
                idx = random.randrange(len(files))
                file_name = files.pop(idx)
            img_path = self._choose_path(class_name, file_name)

        else:
            if self.strategy == 'sequential':
                class_keys = ["__root"] + [k for k in sorted(self.image_files.keys()) if k != "__root"]
                chosen_class = None
                for key in class_keys:
                    files = self.image_files.get(key, [])
                    if files:
                        chosen_class = key
                        break
                if chosen_class is None:
                    return None
                file_name = self.image_files[chosen_class].pop(0)
                img_path = self._choose_path(chosen_class, file_name)
                class_name = chosen_class

            elif self.strategy == 'random':
                available_classes = [k for k, v in self.image_files.items() if v]
                if not available_classes:
                    return None
                class_name = random.choice(available_classes)
                files = self.image_files[class_name]
                idx = random.randrange(len(files))
                file_name = files.pop(idx)
                img_path = self._choose_path(class_name, file_name)

        img = Image.open(img_path)
        if not self.include_alpha and img.mode in ('RGBA', 'LA'):
            img = img.convert('RGB')
        img_array = np.array(img)
        return img_array, file_name, class_name

    def write(self, arr: np.ndarray, file_name: str, class_name,  *args, **kwargs) -> None:
        path = self._choose_path(class_name, file_name, write=True)

        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError(f"File {path} already exists and overwrite is set to False.")

        img = Image.fromarray(arr)
        img.save(path)

    def get_class_names(self):
        return set(self.image_files.keys())
    
    def get_num_entries_by_class(self, class_name: str):
        if class_name not in self.image_files:
            raise ValueError(f"Class name {class_name} is invalid.")
        return len(self.image_files[class_name])
    
    def get_total_samples(self):
        return sum(len(files) for files in self.image_files.values())

    def initialize(self):
        self.image_files = {"__root": []}

        try:
            entries = os.listdir(self.directory_path)
        except FileNotFoundError:
            raise ValueError(f"directory_path {self.directory_path} is not a valid directory.")

        for entry in entries:
            entry_path = os.path.join(self.directory_path, entry)
            if os.path.isdir(entry_path):

                subdirs = [os.path.join(entry_path, d) for d in os.listdir(entry_path)]
                if any(os.path.isdir(p) for p in subdirs):
                    raise ValueError("Nested directories are not supported.")

                os.makedirs(os.path.join(self.output_path, entry), exist_ok=True)

                files = [f for f in os.listdir(entry_path) if f.lower().endswith(self.ALLOWED_EXTS)]
                files.sort()
                self.image_files[entry] = files
            else:
                if entry.lower().endswith(self.ALLOWED_EXTS):
                    self.image_files["__root"].append(entry)

        self.image_files["__root"].sort()

    def get_same_read_write(self):
        return self.same_read_write
