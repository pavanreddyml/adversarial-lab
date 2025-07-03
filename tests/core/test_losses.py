import pytest
import numpy as np
import tensorflow as tf
import torch

from adversarial_lab.core.losses import *

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_loss_instantiation(framework):
    class DummyLoss(Loss):
        def calculate(self, target, predictions, logits, noise, *args, **kwargs):
            pass
    
    loss = DummyLoss()
    loss.set_framework(framework)
    assert loss.framework == framework

def test_dummy_loss_returns_none():
    loss = DummyLoss()
    val = loss.calculate(None, None, None, None)
    assert val is None

def test_loss_invalid_framework():
    class DummyLoss(Loss):
        def calculate(self, target, predictions, logits, *args, **kwargs):
            pass
    
    with pytest.raises(ValueError, match="framework must be either 'tf', 'torch' or 'numpy'"):
        loss = DummyLoss()
        loss.set_framework("invalid")


def test_loss_has_calculate():
    class DummyLoss(Loss):
        def calculate(self, target, predictions, logits, noise, *args, **kwargs):
            pass
    
    loss = DummyLoss()
    assert hasattr(loss, "calculate")

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_binary_cross_entropy(framework):
    loss_fn = BinaryCrossEntropy(from_logits=True)
    loss_fn.set_framework(framework)
    
    if framework == "tf":
        target = tf.constant([0, 1, 1], dtype=tf.float32)
        predictions = tf.constant([0.1, 0.9, 0.8], dtype=tf.float32)
        logits = tf.constant([0.0, 2.0, 1.5], dtype=tf.float32)
        noise = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)
    else:
        target = torch.tensor([0, 1, 1], dtype=torch.float32)
        predictions = torch.tensor([0.1, 0.9, 0.8], dtype=torch.float32)
        logits = torch.tensor([0.0, 2.0, 1.5], dtype=torch.float32)
        noise = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    
    try:
        loss = loss_fn.calculate(target, predictions, logits, noise)
    except NotImplementedError:
        pass
    assert loss_fn.value is not None

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_categorical_cross_entropy(framework):
    loss_fn = CategoricalCrossEntropy(from_logits=True)
    loss_fn.set_framework(framework)
    
    if framework == "tf":
        target = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        predictions = tf.constant([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=tf.float32)
        logits = tf.constant([[0.0, 2.0, -1.0], [1.5, -0.5, -1.0]], dtype=tf.float32)
        noise = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=tf.float32)
    else:
        target = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        predictions = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=torch.float32)
        logits = torch.tensor([[0.0, 2.0, -1.0], [1.5, -0.5, -1.0]], dtype=torch.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    
    try:
        loss = loss_fn.calculate(target, predictions, logits, noise)
    except NotImplementedError:
        pass
    assert loss_fn.value is not None

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_mean_absolute_error(framework):
    loss_fn = MeanAbsoluteError()
    loss_fn.set_framework(framework)
    
    if framework == "tf":
        target = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        predictions = tf.constant([1.5, 1.8, 2.5], dtype=tf.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    else:
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        predictions = torch.tensor([1.5, 1.8, 2.5], dtype=torch.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    
    try:
        loss = loss_fn.calculate(target, predictions, None, noise)
    except NotImplementedError:
        pass
    assert loss_fn.value is not None

@pytest.mark.parametrize("framework", ["torch", "tf"])
def test_mean_squared_error(framework):
    loss_fn = MeanSquaredError()
    loss_fn.set_framework(framework)
    
    if framework == "tf":
        target = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        predictions = tf.constant([1.5, 1.8, 2.5], dtype=tf.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    else:
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        predictions = torch.tensor([1.5, 1.8, 2.5], dtype=torch.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    
    try:
        loss = loss_fn.calculate(target, predictions, None, noise)
    except NotImplementedError:
        pass
    assert loss_fn.value is not None

@pytest.mark.parametrize("function_type", ["valid", "missing_predictions", "missing_logits"])
def test_loss_from_function(function_type):
    if function_type == "missing_predictions":
        def invalid_function(target, logits, from_logits, *args, **kwargs):
            return target

        with pytest.raises(TypeError, match="must have parameter: 'predictions'"):
            LossFromFunction.create(invalid_function)

    if function_type == "missing_logits":
        def invalid_function(target, predictions, from_logits, *args, **kwargs):
            return predictions - target

        with pytest.raises(TypeError, match="must have parameter: 'logits'"):
            LossFromFunction.create(invalid_function)

    if function_type == "valid":
        def valid_function(target, predictions, logits, from_logits, *args, **kwargs):
            return predictions - target

        loss = LossFromFunction.create(valid_function)

        target = tf.constant([1.0, 2.0], dtype=tf.float32)
        predictions = tf.constant([2.0, 3.0], dtype=tf.float32)
        logits = tf.constant([0.0, 0.0], dtype=tf.float32)
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)

        result = loss.calculate(target, predictions, logits, noise)

        assert np.array_equal(result.numpy(), [1.0, 1.0])

@pytest.mark.parametrize("loss_name", ["cce", "bce", "mae", "mse"])
def test_loss_registry(loss_name):
    loss_class = LossRegistry.get(loss_name)
    assert issubclass(loss_class, Loss)