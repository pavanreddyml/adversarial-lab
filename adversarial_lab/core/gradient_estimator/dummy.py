from . import GradientEstimator


class DummyGradientEstimator(GradientEstimator):
    """
    A dummy gradient estimator that does not compute any gradients.
    This is useful for testing purposes or when no gradient estimation is needed.
    """

    def calculate(self, *args, **kwargs):
        """
        Returns None as no gradients are computed.
        """
        return None

    def __repr__(self):
        return "DummyGradientEstimator"