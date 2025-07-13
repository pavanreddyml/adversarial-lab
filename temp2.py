from adversarial_lab.core.gradient_estimator import FDGradientEstimator
import numpy as np

class DummyLoss:
    def calculate(self, target, predictions, logits=None, noise=None):
        # Simple squared error for testing
        return np.sum((predictions - target)**2)

# Instantiate estimator
fdge = FDGradientEstimator(
    epsilon=1e-5,
    max_perturbations=10000,
    batch_size=32,
    block_size=4,
    block_pattern="square"
)

# Setup dummy inputs
np.random.seed(42)
sample = np.random.rand(10, 10)
noise = [np.zeros_like(sample)]

def predict_fn(samples):
    # Return identity predictions, wrapped in a list
    return [s for s in samples]

def construct_target_vector(sample):
    # Return flattened average target
    return np.mean(sample) * np.ones_like(sample)

# Dummy loss instance
loss = DummyLoss()

# Call calculate
grads = fdge.calculate(
    sample=sample,
    noise=noise,
    target_vector=construct_target_vector(sample),
    predict_fn=predict_fn,
    construct_perturbation_fn=lambda x: x[0],
    loss=loss,
    mask=np.ones_like(sample)
)

print(grads[0])
