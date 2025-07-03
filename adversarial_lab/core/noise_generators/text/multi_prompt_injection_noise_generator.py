import numpy as np

from . import TextNoiseGenerator

from typing import Literal, List, Union, Optional, Dict
from adversarial_lab.core.types import TensorType, TensorVariableType
from nltk.tokenize import word_tokenize, sent_tokenize


class PromptInjectionNoiseGenerator(TextNoiseGenerator):
    def __init__(self,
                 position: List[float] | float = 1.0,
                 length: List[int] | int = 10,
                 insertion_strategy: Literal['char', 'word', 'sentence'] = 'word',
                 replacement: Optional[Dict[str, str]] = None,
                 obfuscation: List[Optional[Literal['hex', 'base64']]] | Optional[Literal['hex', 'base64']] = None,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__()

        self.position = position
        self.length = length
        self.insertion_strategy = insertion_strategy

    def generate_noise_meta(self,
                            sample: Union[np.ndarray, TensorType],
                            ) -> TensorVariableType:
        return [' ' * self.length]

    def get_noise(self,
                  noise_meta: List[TensorVariableType]
                  ) -> np.ndarray:
        return np.array([noise_meta[0]])

    def construct_noise(self,
                    noise_meta: List[TensorVariableType]
                    ) -> TensorType:
        return noise_meta[0]

    def apply_noise(self,
                           prompt: str,
                           perturbation: Union[np.ndarray, TensorType],
                           position: float = 1.0
                           ) -> TensorType:
        return self.insert_at_position(prompt, perturbation[0], position)

    def update(self,
               *args,
               **kwargs
               ) -> None:
        pass

    def insert_at_position(self,
                            prompt: str,
                            insertion: Union[np.ndarray, TensorType],
                            position: float
                            ) -> TensorType:
        if self.insertion_strategy == 'char':
            return prompt[:int(position)] + insertion + prompt[int(position):]
        elif self.insertion_strategy == 'word':
            words = word_tokenize(prompt)
            words.insert(int(position), insertion)
            return ' '.join(words)
        elif self.insertion_strategy == 'sentence':
            sentences = sent_tokenize(prompt)
            sentences.insert(int(position), insertion)
            return ' '.join(sentences)
        else:
            raise ValueError(f"Unknown insertion strategy: {self.insertion_strategy}")
