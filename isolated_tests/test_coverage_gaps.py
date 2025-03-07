"""
Tests to cover gaps in the coverage.
"""

import pytest
from typing import Dict, Any, List

from dspy_elo.core.elo_rating import EloRatingSystem
from dspy_elo.core.metric_base import BaseEloMetric
from dspy_elo.core.simple_judge import SimpleJudge

# Use classes instead of dataclasses to avoid pytest collection warnings
class TestExample:
    def __init__(self, question, expected):
        self.question = question
        self.expected = expected

class TestPrediction:
    def __init__(self, answer):
        self.answer = answer

class TestExampleDict(dict):
    def __init__(self, question, expected):
        super().__init__(question=question, expected=expected)

class TestPredictionDict(dict):
    def __init__(self, answer):
        super().__init__(answer=answer)

def test_elo_rating_normalize_all_same():
    """Test normalize_ratings when all ratings are the same."""
    elo = EloRatingSystem()
    
    # Add some ratings that are all the same
    elo.ratings = {"a": 1500.0, "b": 1500.0, "c": 1500.0}
    
    # Normalize ratings
    normalized = elo.normalize_ratings()
    
    # Check that all normalized ratings are the same
    assert all(v == 1.0 for v in normalized.values())
    
    # Test with different target ratio
    normalized = elo.normalize_ratings(target_ratio=0.3)
    assert all(v >= 1.0 for v in normalized.values())
    
    # Test empty ratings case
    elo.ratings = {}
    normalized = elo.normalize_ratings()
    assert normalized == {}
    
    # Test case where all ratings are the same and equal to target_rating
    elo.ratings = {"a": 1500.0, "b": 1500.0}
    normalized = elo.normalize_ratings(target_ratio=0.5)
    assert all(v == 1.0 for v in normalized.values())

def test_simple_judge_custom_compare():
    """Test SimpleJudge with a custom compare function."""
    def custom_compare(question, answer_a, answer_b):
        # Always return A as the winner
        return "A", "Custom compare function", 1.0
    
    judge = SimpleJudge(compare_fn=custom_compare)
    
    # Test the compare function
    winner, explanation, outcome = judge.compare("test", "answer A", "answer B")
    
    assert winner == "A"
    assert explanation == "Custom compare function"
    assert outcome == 1.0
    
    # Test with empty answers
    winner, explanation, outcome = judge.compare("test", "", "")
    assert winner == "A"
    assert explanation == "Custom compare function"
    assert outcome == 1.0

def test_metric_base_edge_cases():
    """Test edge cases in the BaseEloMetric class."""
    # Create a judge function
    def judge_fn(question, answer_a, answer_b):
        return "A", "A is better", 1.0
    
    # Create a metric with custom min/max scores
    metric = BaseEloMetric(
        judge_fn=judge_fn,
        min_score=0.0,
        max_score=10.0
    )
    
    # Test with trace parameter
    example = TestExample(question="What is 2+2?", expected="4")
    prediction = TestPrediction(answer="4")
    
    # First prediction should return the default normalized score with trace
    result = metric(example, prediction, trace=True)
    assert isinstance(result, bool)
    
    # Add more predictions to test comparison
    for i in range(10):
        metric(example, TestPrediction(answer=f"{i}"))
    
    # Test normalize_score with string rating
    score = metric.normalize_score("pred_0")
    assert 0.0 <= score <= 10.0
    
    # Test normalize_score with float rating
    # Test extremely low rating
    score = metric.normalize_score(metric.elo_system.default_rating - 1500)
    assert score == 0.0
    
    # Test extremely high rating
    score = metric.normalize_score(metric.elo_system.default_rating + 1500)
    assert score == 10.0
    
    # Test rating that doesn't match any prediction
    score = metric.normalize_score(9999.0)
    assert 0.0 <= score <= 10.0
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        BaseEloMetric(judge_fn=judge_fn, num_comparisons=-1)
    
    with pytest.raises(ValueError):
        BaseEloMetric(judge_fn=judge_fn, normalize_ratio=-0.1)
    
    with pytest.raises(ValueError):
        BaseEloMetric(judge_fn=judge_fn, normalize_ratio=1.1)
    
    with pytest.raises(ValueError):
        BaseEloMetric(judge_fn=judge_fn, min_score=10.0, max_score=5.0)

def test_metric_base_extract_field_edge_cases():
    """Test edge cases in the _extract_field method."""
    def judge_fn(question, answer_a, answer_b):
        return "A", "A is better", 1.0
    
    metric = BaseEloMetric(judge_fn=judge_fn)
    
    # Test with None object
    result = metric._extract_field(None, "field", "default")
    assert result == "default"
    
    # Test with object that raises TypeError on getattr
    class BadObject:
        def __getattribute__(self, name):
            raise TypeError("Bad getattr")
    
    result = metric._extract_field(BadObject(), "field", "default")
    assert result == "default"
    
    # Test with object that raises TypeError on __getitem__
    class BadDict:
        def __getitem__(self, key):
            raise TypeError("Bad getitem")
    
    result = metric._extract_field(BadDict(), "field", "default")
    assert result == "default"

def test_metric_base_call_with_trace():
    """Test the __call__ method with trace parameter."""
    def judge_fn(question, answer_a, answer_b):
        return "A", "A is better", 1.0
    
    metric = BaseEloMetric(judge_fn=judge_fn)
    
    # Add some predictions
    example = TestExample(question="What is 2+2?", expected="4")
    
    # First prediction
    prediction1 = TestPrediction(answer="4")
    result1 = metric(example, prediction1)
    
    # Second prediction with trace
    prediction2 = TestPrediction(answer="5")
    result2 = metric(example, prediction2, trace=True)
    assert isinstance(result2, bool)
    
    # Third prediction with different question
    example2 = TestExample(question="What is 3+3?", expected="6")
    prediction3 = TestPrediction(answer="6")
    result3 = metric(example2, prediction3)
    
    # Test with same question but different prediction
    result4 = metric(example, TestPrediction(answer="Different answer"), trace=True)
    assert isinstance(result4, bool)

if __name__ == "__main__":
    print("Running tests directly...")
    
    # Run the tests directly
    test_elo_rating_normalize_all_same()
    test_simple_judge_custom_compare()
    test_metric_base_edge_cases()
    test_metric_base_extract_field_edge_cases()
    test_metric_base_call_with_trace()
    
    print("All tests completed successfully!")
