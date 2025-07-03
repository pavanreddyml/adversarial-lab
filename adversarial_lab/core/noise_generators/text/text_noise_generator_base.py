from abc import ABC, abstractmethod, ABCMeta

import base64
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from adversarial_lab.core.noise_generators import NoiseGenerator
from adversarial_lab.core.tensor_ops import TensorOps
from adversarial_lab.core.optimizers import Optimizer

from typing import Literal, Union, List, Tuple, Optional, Callable, Dict
from adversarial_lab.core.types import TensorType, TensorVariableType, OptimizerType


class TextNoiseGenerator(NoiseGenerator):
    def __init__(self) -> None:
        pass

    def _replace_text(self, text: str, replacement: Dict[str, str]) -> str:
        for key, value in replacement.items():
            text = text.replace(key, value)
        return text

    def _obfuscate_text(self, text: str, obfuscation: Optional[Literal['hex', 'base64']] = None) -> str:
        if obfuscation == 'hex':
            return text.encode('utf-8').hex()
        elif obfuscation == 'base64':
            return base64.b64encode(text.encode('utf-8')).decode('utf-8')
        elif obfuscation == 'ascii':
            return text.encode('ascii', 'ignore').decode('ascii')
        else:
            return text
        
    def _insert_at_position(self,
                        prompt: str,
                        noise: List[Dict[str, Union[str, float]]],
                        ) -> str:
        noise.sort(key=lambda x: x['position'])
        insertion_type = noise[0]['insertion']

        if insertion_type == 'char':
            text = ""
            noise_idx = 0
            for i, char in enumerate(prompt):
                if noise_idx < len(noise) and i / len(prompt) >= noise[noise_idx]["position"]:
                    text += noise[noise_idx]['text']
                    noise_idx += 1
                text += char
            return text

        elif insertion_type == 'word':
            from nltk.tokenize import word_tokenize
            words = word_tokenize(prompt)
            text = ""
            noise_idx = 0
            for i, word in enumerate(words):
                if noise_idx < len(noise) and i / len(words) >= noise[noise_idx]["position"]:
                    text += noise[noise_idx]['text'] + ' '
                    noise_idx += 1
                text += word + ' '
            return text.strip()

        elif insertion_type == 'sentence':
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(prompt)
            text = ""
            noise_idx = 0
            for i, sentence in enumerate(sentences):
                if noise_idx < len(noise) and i / len(sentences) >= noise[noise_idx]["position"]:
                    text += noise[noise_idx]['text'] + ' '
                    noise_idx += 1
                text += sentence + ' '
            return text.strip()

        else:
            raise ValueError(f"Unknown insertion strategy: {insertion_type}")

