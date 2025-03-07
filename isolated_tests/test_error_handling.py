"""
Tests for error handling in the Elo metric.
"""

import pytest
from typing import Dict, Any, List

from dspy_elo.core.metric_base import BaseEloMetric
from isolated_tests.mock_classes_no_dataclass import (
    MockExample, MockPrediction, MockExampleDict, MockPredictionDict, MockEloMetric
)

def test_none_inputs():
    """Test handling of None inputs."""
    metric = MockEloMetric()
    
    # Test with None example
    with pytest.raises(ValueError, match="Example cannot be None"):
        metric(None, MockPrediction(answer="test"))
    
    # Test with None prediction
    with pytest.raises(ValueError, match="Prediction cannot be None"):
        metric(MockExample(question="test", expected="test"), None)

def test_empty_inputs():
    """Test handling of empty inputs."""
    metric = MockEloMetric()
    
    # Empty inputs should work, just not be very useful
    result = metric(MockExample(question="", expected=""), MockPrediction(answer=""))
    assert result is not None

def test_invalid_field_names():
    """Test handling of invalid field names."""
    # Test with invalid question field
    metric = MockEloMetric(question_field="invalid_field")
    example = MockExample(question="test", expected="test")
    prediction = MockPrediction(answer="test")
    
    # This should work, but use the string representation of the example
    result = metric(example, prediction)
    assert result is not None
    
    # Test with invalid answer field
    metric = MockEloMetric(answer_field="invalid_field")
    
    # This should work, but use the string representation of the prediction
    result = metric(example, prediction)
    assert result is not None

def test_invalid_parameters():
    """Test handling of invalid parameters."""
    # Test with negative num_comparisons
    with pytest.raises(ValueError, match="num_comparisons must be non-negative"):
        MockEloMetric(num_comparisons=-1)
    
    # Test with normalize_ratio < 0
    with pytest.raises(ValueError, match="normalize_ratio must be between 0 and 1"):
        MockEloMetric(normalize_ratio=-0.1)
    
    # Test with normalize_ratio > 1
    with pytest.raises(ValueError, match="normalize_ratio must be between 0 and 1"):
        MockEloMetric(normalize_ratio=1.1)
    
    # Test with min_score > max_score
    with pytest.raises(ValueError, match="min_score must be less than or equal to max_score"):
        MockEloMetric(min_score=1.0, max_score=0.5)

def test_mixed_input_types():
    """Test handling of mixed input types."""
    metric = MockEloMetric()
    
    # Test with different example types
    example1 = MockExample(question="test", expected="test")
    example2 = MockExampleDict(question="test", expected="test")
    example3 = {"question": "test", "expected": "test"}
    
    # Test with different prediction types
    prediction1 = MockPrediction(answer="test")
    prediction2 = MockPredictionDict(answer="test")
    prediction3 = {"answer": "test"}
    
    # All combinations should work
    for example in [example1, example2, example3]:
        for prediction in [prediction1, prediction2, prediction3]:
            result = metric(example, prediction)
            assert result is not None

def test_trace_parameter():
    """Test handling of trace parameter."""
    metric = MockEloMetric()
    
    # Test with trace=None (default)
    result1 = metric(MockExample(question="test", expected="test"), MockPrediction(answer="test"))
    assert isinstance(result1, float)
    
    # Test with trace=True
    result2 = metric(MockExample(question="test", expected="test"), MockPrediction(answer="test"), trace=True)
    assert isinstance(result2, bool)
    
    # Test with trace=False
    result3 = metric(MockExample(question="test", expected="test"), MockPrediction(answer="test"), trace=False)
    assert isinstance(result3, bool)
    
    # Test with trace={}
    result4 = metric(MockExample(question="test", expected="test"), MockPrediction(answer="test"), trace={})
    assert isinstance(result4, bool)

def test_edge_case_ratings():
    """Test handling of edge case ratings."""
    # Create a simple judge function
    def judge_fn(question, answer_a, answer_b):
        return "A", "A is better", 1.0
    
    # Use the actual BaseEloMetric class for this test
    metric = BaseEloMetric(
        judge_fn=judge_fn,
        min_score=0.0,
        max_score=2.0
    )
    
    # Test with extremely low rating
    score = metric.normalize_score(metric.elo_system.default_rating - 1500)
    assert metric.min_score <= score <= metric.max_score
    
    # Add a prediction to make sure we have at least one rating
    example = MockExample(question="test", expected="test")
    prediction = MockPrediction(answer="test")
    metric(example, prediction)
    
    # Now test with extremely low rating again
    score = metric.normalize_score(metric.elo_system.default_rating - 1500)
    assert metric.min_score <= score <= metric.max_score
    
    # Test with extremely high rating
    score = metric.normalize_score(metric.elo_system.default_rating + 1500)
    assert metric.min_score <= score <= metric.max_score

def test_custom_min_max_scores():
    """Test handling of custom min/max scores."""
    # Create a simple judge function
    def judge_fn(question, answer_a, answer_b):
        return "A", "A is better", 1.0
    
    # Use the actual BaseEloMetric class for this test
    metric = BaseEloMetric(
        judge_fn=judge_fn,
        min_score=-10.0,
        max_score=10.0
    )
    
    # First prediction should return the default normalized score
    example = MockExample(question="test", expected="test")
    prediction = MockPrediction(answer="test")
    
    result = metric(example, prediction)
    assert -10.0 <= result <= 10.0
    
    # Test normalize_score with custom min/max
    score = metric.normalize_score(metric.elo_system.default_rating)
    assert -10.0 <= score <= 10.0

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
