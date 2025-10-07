from abc import ABC, abstractmethod


class BasePreprocessingDefense(ABC):
    @abstractmethod
    def run(self, data):
        pass

    def __call__(self, data):
        return self.preprocess(data)