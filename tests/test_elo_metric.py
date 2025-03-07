"""
Tests for the EloMetric class.
"""

import pytest
from unittest.mock import MagicMock, patch
import uuid
import dspy
from dataclasses import dataclass
from dspy_elo.elo_metric import EloMetric


# Create a simple example and prediction class for testing
@dataclass
class TestExample:
    question: str
    answer: str = None


@dataclass
class TestPrediction:
    answer: str


@pytest.fixture
def mock_judge():
    """Create a mock judge."""
    mock = MagicMock()
    # Default behavior: return A as winner
    mock.compare.return_value = ("A", "A is better", 1.0)
    return mock


@pytest.fixture
def elo_metric(mock_judge):
    """Create an EloMetric instance with a mock judge."""
    metric = EloMetric(
        criteria="Test criteria",
        num_comparisons=2,
        normalize_ratio=0.5
    )
    metric.judge = mock_judge
    return metric


def test_elo_metric_init():
    """Test EloMetric initialization."""
    criteria = "Test criteria"
    metric = EloMetric(
        criteria=criteria,
        num_comparisons=3,
        normalize_ratio=0.6,
        default_rating=2000,
        k_factor=24
    )
    
    assert metric.criteria == criteria
    assert metric.num_comparisons == 3
    assert metric.normalize_ratio == 0.6
    assert metric.elo_system.default_rating == 2000
    assert metric.elo_system.k_factor == 24
    assert isinstance(metric.judge, dspy.Module)
    assert len(metric.examples) == 0
    assert len(metric.predictions) == 0


def test_first_prediction(elo_metric):
    """Test evaluating the first prediction."""
    # Create example and prediction
    example = TestExample(question="Test question")
    prediction = TestPrediction(answer="Test answer")
    
    # Evaluate
    score = elo_metric(example, prediction)
    
    # First prediction should get the default normalized score
    expected_score = elo_metric.normalize_score(elo_metric.elo_system.default_rating)
    assert score == expected_score
    
    # Should store the example and prediction
    assert len(elo_metric.examples) == 1
    assert len(elo_metric.predictions) == 1


def test_multiple_predictions(elo_metric, monkeypatch):
    """Test evaluating multiple predictions."""
    # Mock uuid.uuid4 to return predictable values
    mock_uuids = ["id1", "id2", "id3"]
    mock_uuid_gen = MagicMock()
    mock_uuid_gen.side_effect = mock_uuids
    monkeypatch.setattr(uuid, "uuid4", mock_uuid_gen)
    
    # Create examples and predictions
    examples = [
        TestExample(question="Question 1"),
        TestExample(question="Question 1"),
        TestExample(question="Question 2")
    ]
    
    predictions = [
        TestPrediction(answer="Answer 1"),
        TestPrediction(answer="Answer 2"),
        TestPrediction(answer="Answer 3")
    ]
    
    # First prediction gets default score
    score1 = elo_metric(examples[0], predictions[0])
    
    # Second prediction should be compared to first
    # Mock judge returns A as winner, so second prediction should win
    score2 = elo_metric(examples[1], predictions[1])
    
    # Third prediction has a different question, so it shouldn't be compared
    score3 = elo_metric(examples[2], predictions[2])
    
    # Check that we stored all examples and predictions
    assert len(elo_metric.examples) == 3
    assert len(elo_metric.predictions) == 3
    
    # Check that ratings were updated
    assert elo_metric.elo_system.get_rating("id1") < elo_metric.elo_system.default_rating
    assert elo_metric.elo_system.get_rating("id2") > elo_metric.elo_system.default_rating
    
    # Check that history was recorded
    assert len(elo_metric.elo_system.history) == 1
    assert elo_metric.elo_system.history[0]["item_a"] == "id2"
    assert elo_metric.elo_system.history[0]["item_b"] == "id1"


def test_select_comparison_samples(elo_metric, monkeypatch):
    """Test _select_comparison_samples method."""
    # Mock find_closest_ratings to return predictable values
    elo_metric.elo_system.find_closest_ratings = MagicMock(return_value=["id1", "id2"])
    
    # Add some predictions
    elo_metric.predictions = {
        "id1": MagicMock(),
        "id2": MagicMock(),
        "id3": MagicMock()
    }
    
    # Select samples
    samples = elo_metric._select_comparison_samples("id3")
    
    # Should return what find_closest_ratings returns
    assert samples == ["id1", "id2"]
    elo_metric.elo_system.find_closest_ratings.assert_called_once_with("id3", 2)
    
    # When find_closest_ratings returns empty list, should randomly select
    elo_metric.elo_system.find_closest_ratings.return_value = []
    samples = elo_metric._select_comparison_samples("id3")
    assert len(samples) == 2
    assert all(s in ["id1", "id2"] for s in samples)


def test_normalize_score(elo_metric):
    """Test normalize_score method."""
    # Add some ratings
    elo_metric.predictions = {
        "id1": MagicMock(),
        "id2": MagicMock(),
        "id3": MagicMock()
    }
    elo_metric.elo_system.ratings = {
        "id1": 1400,
        "id2": 1500,
        "id3": 1600
    }
    
    # Mock normalize_ratings
    elo_metric.elo_system.normalize_ratings = MagicMock(return_value={
        "id1": 0.8,
        "id2": 1.0,
        "id3": 1.2
    })
    
    # Normalize scores
    assert elo_metric.normalize_score("id1") == 0.8
    assert elo_metric.normalize_score("id2") == 1.0
    assert elo_metric.normalize_score("id3") == 1.2
    
    # Should clip to min_score and max_score
    elo_metric.min_score = 0.5
    elo_metric.max_score = 1.1
    assert elo_metric.normalize_score("id1") == 0.8
    assert elo_metric.normalize_score("id3") == 1.1  # Clipped to max_score


def test_trace_handling(elo_metric):
    """Test handling of trace parameter."""
    # Create example and prediction
    example = TestExample(question="Test question")
    prediction = TestPrediction(answer="Test answer")
    
    # With trace=None, should return float
    score = elo_metric(example, prediction)
    assert isinstance(score, float)
    
    # With trace not None, should return bool
    trace = [MagicMock()]
    result = elo_metric(example, prediction, trace=trace)
    assert isinstance(result, bool)
