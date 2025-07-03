import numpy as np

from . import TextNoiseGenerator

from typing import Literal, List, Union, Optional, Dict
from nltk.tokenize import word_tokenize, sent_tokenize


class PromptInjectionNoiseGenerator(TextNoiseGenerator):
    def __init__(self,
                 position: float = 1.0,
                 length: int = 10,
                 insertion: Literal['char',
                                    'word', 'sentence'] = 'word',
                 replacement: Optional[Dict[str, str]] = None,
                 obfuscation: Optional[Literal['hex', 'base64', 'ascii']] = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__()

        if not isinstance(position, float):
            raise TypeError(
                f"Expected position to be float, got {type(position)}")
        if position < 0 or position > 1:
            raise ValueError(
                f"Position must be between 0 and 1, got {position}")

        if not isinstance(length, int):
            raise TypeError(f"Expected length to be int, got {type(length)}")

        if insertion not in ['char', 'word', 'sentence']:
            raise ValueError(
                f"Unknown insertion strategy: {insertion}")

        if replacement is not None and not isinstance(replacement, dict):
            raise TypeError(
                f"Expected replacement to be dict, got {type(replacement)}")

        if obfuscation is not None and obfuscation not in [None, 'hex', 'base64', 'ascii']:
            raise ValueError(f"Unknown obfuscation type: {obfuscation}")

        self.position = position
        self.length = length
        self.insertion = insertion
        self.replacement = replacement or {}
        self.obfuscation = obfuscation

    def generate_noise_meta(self,
                            sample: str,
                            ) -> List[str]:
        pass

    def get_noise(self,
                  noise_meta: List[str]
                  ) -> List[str]:
        pass

    def construct_noise(self,
                        noise_meta: List[str]
                        ) -> str:
        pass

    def apply_noise(self,
                    prompt: str,
                    noise
                    ) -> str:
        pass

    def update(self,
               *args,
               **kwargs
               ) -> None:
        pass

