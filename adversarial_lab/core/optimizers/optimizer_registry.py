from . import Adam, SGD, PGD, Optimizer

class OptimizerRegistry:
    __optmizers = {
        "adam": Adam,
        "sgd": SGD,
        "pgd": PGD,
    }

    @classmethod
    def get(cls, optimizer: str) -> Optimizer:
        """
        Retrieve an optimizer class by name.

        This method returns the corresponding optimizer class based on the provided string 
        identifier. If the optimizer name is not found in the registry, a `ValueError` is raised.

        Parameters:
            optimizer (str): The name of the optimizer as a string.

        Returns:
            Type[Optimizer]: The corresponding optimizer class.

        Raises:
            ValueError: If the provided optimizer name does not exist in the registry.

        Notes:
            - Supported optimizer names: `"adam"` (Adam optimizer), `"sgd"` (Stochastic Gradient Descent),
            `"pgd"` (Projected Gradient Descent).
            - The method ensures that only registered optimizers can be retrieved.
            - If an invalid optimizer name is provided, a `ValueError` is raised.
        """
        if isinstance(optimizer, Optimizer):
            return optimizer
        if optimizer not in cls.__optmizers:
            raise ValueError(f"Invalid value for optmizer: '{optimizer}'. Optimizer of this type does not exist.")
        return cls.__optmizers[optimizer]
