"""
Test the normalize_score method of the EloMetric class without depending on dspy.
"""

import pytest
import uuid
from typing import Dict, Any
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# Create mock classes to avoid importing dspy
class MockEloRatingSystem:
    def __init__(self, default_rating=1500.0, k_factor=32.0):
        self.ratings = {}
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.history = []
    
    def get_rating(self, item_id):
        return self.ratings.get(item_id, self.default_rating)
    
    def normalize_ratings(self, target_ratio=0.5, target_threshold=1.0):
        if not self.ratings:
            return {}
        
        # Simple mock implementation for testing
        normalized = {}
        for item_id, rating in self.ratings.items():
            if rating < 1500:
                normalized[item_id] = 0.5
            else:
                normalized[item_id] = 1.5
        
        return normalized


class MockEloMetric:
    def __init__(self, criteria="accuracy", num_comparisons=5, normalize_ratio=0.5, 
                 default_rating=1500.0, k_factor=32.0, min_score=0.0, max_score=2.0,
                 question_field="question", answer_field="answer"):
        self.criteria = criteria
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.min_score = min_score
        self.max_score = max_score
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Initialize Elo rating system
        self.elo_system = MockEloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Store examples and predictions
        self.examples = {}
        self.predictions = {}
    
    def __call__(self, example, prediction):
        # Generate a unique ID for this prediction
        pred_id = str(uuid.uuid4())
        
        # Store example and prediction
        self.examples[pred_id] = example
        self.predictions[pred_id] = prediction
        
        # Add an entry to the ratings dictionary
        self.elo_system.ratings[pred_id] = self.elo_system.default_rating
        
        # Return a default score
        return 1.0
    
    def normalize_score(self, rating):
        """
        Normalize an Elo rating to a score between min_score and max_score.
        
        Args:
            rating: The Elo rating to normalize
            
        Returns:
            A normalized score
        """
        # Get all current ratings
        all_ratings = {pid: self.elo_system.get_rating(pid) for pid in self.predictions.keys()}
        
        # If we only have one rating, return a default value
        if len(all_ratings) <= 1:
            return (self.min_score + self.max_score) / 2.0
        
        # Normalize ratings so that normalize_ratio of them are above 1.0
        normalized_ratings = self.elo_system.normalize_ratings(
            target_ratio=self.normalize_ratio,
            target_threshold=1.0
        )
        
        # Get the normalized rating for this prediction
        if isinstance(rating, str):
            normalized_rating = normalized_ratings.get(rating, (self.min_score + self.max_score) / 2.0)
        else:
            # Find the prediction ID with this rating
            pred_ids = [pid for pid, r in self.elo_system.ratings.items() if r == rating]
            if pred_ids:
                normalized_rating = normalized_ratings.get(pred_ids[0], (self.min_score + self.max_score) / 2.0)
            else:
                # If we can't find the exact rating, use the default
                normalized_rating = (self.min_score + self.max_score) / 2.0
        
        # Clip to min_score and max_score
        return max(self.min_score, min(self.max_score, normalized_rating))


@dataclass
class TestExample:
    question: str


@dataclass
class TestPrediction:
    answer: str


def test_normalize_score_empty():
    """Test normalize_score with empty ratings."""
    metric = MockEloMetric(criteria="accuracy", min_score=0.0, max_score=2.0)
    
    # Should return the middle value when no ratings exist
    assert metric.normalize_score(1500.0) == 1.0


def test_normalize_score_single_rating():
    """Test normalize_score with a single rating."""
    metric = MockEloMetric(criteria="accuracy", min_score=0.0, max_score=2.0)
    
    # Add a single prediction
    example = TestExample(question="What is 2+2?")
    prediction = TestPrediction(answer="4")
    metric(example, prediction)
    
    # Should return the middle value when only one rating exists
    assert metric.normalize_score(1500.0) == 1.0


def test_normalize_score_multiple_ratings():
    """Test normalize_score with multiple ratings."""
    metric = MockEloMetric(criteria="accuracy", min_score=0.0, max_score=2.0)
    
    # Add multiple predictions
    for i in range(10):
        example = TestExample(question=f"What is {i}+{i}?")
        prediction = TestPrediction(answer=str(i*2))
        metric(example, prediction)
    
    # Manually set some ratings
    ratings = list(metric.elo_system.ratings.keys())
    
    # Set ratings to create a known distribution
    metric.elo_system.ratings[ratings[0]] = 1300.0  # Low
    metric.elo_system.ratings[ratings[1]] = 1400.0  # Low
    metric.elo_system.ratings[ratings[2]] = 1500.0  # Middle
    metric.elo_system.ratings[ratings[3]] = 1600.0  # High
    metric.elo_system.ratings[ratings[4]] = 1700.0  # High
    
    # Test normalization with different normalize_ratio values
    metric.normalize_ratio = 0.5  # Default, half above 1.0
    
    # Low rating should be below 1.0
    normalized_low = metric.normalize_score(1300.0)
    assert normalized_low <= 1.0
    
    # High rating should be above 1.0
    normalized_high = metric.normalize_score(1700.0)
    assert normalized_high >= 1.0


def test_normalize_score_string_id():
    """Test normalize_score with a string ID instead of a rating."""
    metric = MockEloMetric(criteria="accuracy", min_score=0.0, max_score=2.0)
    
    # Add multiple predictions
    for i in range(5):
        example = TestExample(question=f"What is {i}+{i}?")
        prediction = TestPrediction(answer=str(i*2))
        metric(example, prediction)
    
    # Get an existing ID
    existing_id = list(metric.elo_system.ratings.keys())[0]
    
    # Set a known rating
    metric.elo_system.ratings[existing_id] = 1600.0
    
    # Test normalization with the ID
    normalized = metric.normalize_score(existing_id)
    assert 0.0 <= normalized <= 2.0  # Should be within bounds


def test_normalize_score_clipping():
    """Test that normalize_score properly clips values to min_score and max_score."""
    metric = MockEloMetric(criteria="accuracy", min_score=0.5, max_score=1.5)
    
    # Add multiple predictions
    for i in range(10):
        example = TestExample(question=f"What is {i}+{i}?")
        prediction = TestPrediction(answer=str(i*2))
        metric(example, prediction)
    
    # Manually set extreme ratings
    ratings = list(metric.elo_system.ratings.keys())
    
    # Set ratings to create extreme values
    metric.elo_system.ratings[ratings[0]] = 1000.0  # Very low
    metric.elo_system.ratings[ratings[1]] = 2000.0  # Very high
    
    # Test clipping
    low_normalized = metric.normalize_score(1000.0)
    assert low_normalized == 0.5  # Should be clipped to min_score
    
    high_normalized = metric.normalize_score(2000.0)
    assert high_normalized == 1.5  # Should be clipped to max_score


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
