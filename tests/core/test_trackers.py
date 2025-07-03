import numpy as np
import pytest

from adversarial_lab.analytics import ImageTracker, PredictionsTracker

@pytest.fixture
def image_tracker():
    return ImageTracker()

@pytest.fixture
def predictions_tracker():
    return PredictionsTracker(strategy="first_topk", topk=1)

def test_tracker_columns(image_tracker):
    assert isinstance(image_tracker._columns, dict)

def test_predictions_tracker_index_single(predictions_tracker):
    preds = np.array([0.1, 0.8, 0.3])
    predictions_tracker._initialize_index(preds)
    assert predictions_tracker.indexes.count(1) == 1

def test_predictions_tracker_index_batch(predictions_tracker):
    predictions_tracker.indexes = None
    preds_batch = np.array([[0.1, 0.2, 0.3]])
    predictions_tracker._initialize_index(preds_batch)
    assert len(predictions_tracker.indexes) == 3
