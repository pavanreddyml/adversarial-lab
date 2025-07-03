from adversarial_lab.core.optimizers import Optimizer


class PGD(Optimizer):
    """
    Projected Gradient Descent (PGD) optimizer class for adversarial training.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 projection_fn=None,
                 ) -> None:
        """
        Initialize the PGD optimizer with the specified hyperparameters.

        Args:
            learning_rate (float): The step size used for updating weights. Default is 0.001.
            projection_fn (callable, optional): A function that projects updated 
                parameters onto a feasible set to enforce constraints.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.projection_fn = projection_fn

    def initialize_optimizer(self):
        self.optimizer = self.tensor_ops.optimizers.PGD(
            learning_rate=self.learning_rate,
            projection_fn=self.projection_fn,
        )

