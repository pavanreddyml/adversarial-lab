from typing import Callable
from . import Penalty
import inspect

class PenaltyFromFunction:
    @staticmethod
    def create(function: Callable):
        sig = inspect.signature(function)
        params = sig.parameters
        
        has_noise = 'noise' in params
        has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if not has_noise:
            raise TypeError("Function to create POConstraint must have parameter: 'noise'.")

        if not has_args:
            raise TypeError("Function to create POConstraint must have parameter: '*args'.")
        
        if not has_kwargs:
            raise TypeError("Function to create POConstraint must have parameter: '**kwargs'.")
        
        class CustomPenaltyFromFunction(Penalty):
            def __init__(self, func: Callable):
                self.function = func

            def calculate(self, noise, *args, **kwargs):
                return self.function(noise)

        return CustomPenaltyFromFunction(function)
    
    def __call__(self, function: Callable, *args, **kwds):
        return self.create(function, *args, **kwds)