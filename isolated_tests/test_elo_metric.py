"""
Isolated tests for the EloMetric class.
"""

import pytest
from unittest.mock import MagicMock, patch
import uuid
import sys
import os
from dataclasses import dataclass

# Add the parent directory to the path so we can import from fully_isolated_test_elo_metric_direct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the isolated implementation
from fully_isolated_test_elo_metric_direct import SimpleEloMetric


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
    metric = SimpleEloMetric(
        criteria="Test criteria",
        num_comparisons=2,
        normalize_ratio=0.5
    )
    metric.judge = mock_judge
    return metric


def test_elo_metric_init():
    """Test EloMetric initialization."""
    criteria = "Test criteria"
    metric = SimpleEloMetric(
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
    assert len(metric.examples) == 0
    assert len(metric.predictions) == 0


def test_first_prediction(elo_metric=None):
    """Test evaluating the first prediction."""
    # If elo_metric is not provided (when running via run_all_isolated_tests.py),
    # create one with a mock judge
    if elo_metric is None:
        mock_judge = MagicMock()
        mock_judge.compare.return_value = ("A", "A is better", 1.0)
        elo_metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
        elo_metric.judge = mock_judge
    
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


def test_multiple_predictions(elo_metric=None, monkeypatch=None):
    """Test evaluating multiple predictions."""
    # If elo_metric is not provided (when running via run_all_isolated_tests.py),
    # create one with a mock judge
    if elo_metric is None:
        mock_judge = MagicMock()
        mock_judge.compare.return_value = ("A", "A is better", 1.0)
        elo_metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
        elo_metric.judge = mock_judge
    
    # Mock uuid.uuid4 to return predictable values
    mock_uuids = ["id1", "id2", "id3"]
    original_uuid4 = uuid.uuid4
    
    # If monkeypatch is provided, use it, otherwise manually patch
    if monkeypatch:
        mock_uuid_gen = MagicMock()
        mock_uuid_gen.side_effect = mock_uuids
        monkeypatch.setattr(uuid, "uuid4", mock_uuid_gen)
    else:
        # Manual patching
        uuid.uuid4 = MagicMock(side_effect=mock_uuids)
    
    try:
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
    finally:
        # Restore original uuid4 if we manually patched
        if not monkeypatch:
            uuid.uuid4 = original_uuid4


def test_select_comparison_samples(elo_metric=None, monkeypatch=None):
    """Test _select_comparison_samples method."""
    # If elo_metric is not provided (when running via run_all_isolated_tests.py),
    # create one with a mock judge
    if elo_metric is None:
        mock_judge = MagicMock()
        mock_judge.compare.return_value = ("A", "A is better", 1.0)
        elo_metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
        elo_metric.judge = mock_judge
    
    # Save original method
    original_find_closest_ratings = elo_metric.elo_system.find_closest_ratings
    
    try:
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
    finally:
        # Restore original method if we manually mocked it
        if not isinstance(original_find_closest_ratings, MagicMock):
            elo_metric.elo_system.find_closest_ratings = original_find_closest_ratings


def test_normalize_score(elo_metric=None):
    """Test normalize_score method."""
    # If elo_metric is not provided (when running via run_all_isolated_tests.py),
    # create one with a mock judge
    if elo_metric is None:
        mock_judge = MagicMock()
        mock_judge.compare.return_value = ("A", "A is better", 1.0)
        elo_metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
        elo_metric.judge = mock_judge
    
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
    original_normalize_ratings = elo_metric.elo_system.normalize_ratings
    elo_metric.elo_system.normalize_ratings = MagicMock(return_value={
        "id1": 0.8,
        "id2": 1.0,
        "id3": 1.2
    })
    
    try:
        # Normalize scores
        assert elo_metric.normalize_score("id1") == 0.8
        assert elo_metric.normalize_score("id2") == 1.0
        assert elo_metric.normalize_score("id3") == 1.2
        
        # Should clip to min_score and max_score
        elo_metric.min_score = 0.5
        elo_metric.max_score = 1.1
        assert elo_metric.normalize_score("id1") == 0.8
        assert elo_metric.normalize_score("id3") == 1.1  # Clipped to max_score
    finally:
        # Restore original normalize_ratings if we manually mocked it
        if not isinstance(original_normalize_ratings, MagicMock):
            elo_metric.elo_system.normalize_ratings = original_normalize_ratings
