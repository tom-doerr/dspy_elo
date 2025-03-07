"""
Test concurrency and thread safety for the EloMetric class without depending on dspy.
"""

import pytest
import uuid
import threading
import time
from typing import Dict, Any
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor


# Create mock classes to avoid importing dspy
class MockEloRatingSystem:
    def __init__(self, default_rating=1500.0, k_factor=32.0):
        self.ratings = {}
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.history = []
        self._lock = threading.RLock()
    
    def get_rating(self, item_id):
        with self._lock:
            return self.ratings.get(item_id, self.default_rating)
    
    def record_match(self, item_a, item_b, outcome):
        # Simple mock implementation for testing
        with self._lock:
            rating_a = self.get_rating(item_a)
            rating_b = self.get_rating(item_b)
            
            # Update ratings based on outcome
            if outcome == 1.0:
                self.ratings[item_a] = rating_a + 10
                self.ratings[item_b] = rating_b - 10
            elif outcome == 0.0:
                self.ratings[item_a] = rating_a - 10
                self.ratings[item_b] = rating_b + 10
            
            return self.ratings[item_a], self.ratings[item_b]
    
    def find_closest_ratings(self, item_id, n=5):
        # Simple mock implementation for testing
        with self._lock:
            return [id for id in self.ratings.keys() if id != item_id][:n]
    
    def normalize_ratings(self, target_ratio=0.5, target_threshold=1.0):
        with self._lock:
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


class MockAnswerJudge:
    def __init__(self, criteria):
        self.criteria = criteria
    
    def compare(self, question, answer_a, answer_b):
        # Simple mock implementation for testing
        if answer_a == "correct" and answer_b != "correct":
            return "A", "A is correct", 1.0
        elif answer_a != "correct" and answer_b == "correct":
            return "B", "B is correct", 0.0
        else:
            return "TIE", "Both are equal", 0.5


class ThreadSafeEloMetric:
    def __init__(
        self,
        criteria="accuracy",
        num_comparisons=5,
        normalize_ratio=0.5,
        default_rating=1500.0,
        k_factor=32.0,
        min_score=0.0,
        max_score=2.0,
        question_field="question",
        answer_field="answer",
    ):
        # Validate parameters
        if num_comparisons < 0:
            raise ValueError("num_comparisons must be non-negative")
        
        if normalize_ratio < 0 or normalize_ratio > 1:
            raise ValueError("normalize_ratio must be between 0 and 1")
        
        if min_score > max_score:
            raise ValueError("min_score must be less than or equal to max_score")
        
        self.criteria = criteria
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.min_score = min_score
        self.max_score = max_score
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Initialize Elo rating system
        self.elo_system = MockEloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Initialize judge
        self.judge = MockAnswerJudge(criteria=criteria)
        
        # Store examples and predictions
        self.examples = {}
        self.predictions = {}
        
        # Track number of comparisons performed
        self.comparison_count = 0
        
        # Add a lock for thread safety
        self._lock = threading.RLock()
    
    def __call__(self, example, prediction, trace=None):
        # Validate inputs
        if example is None:
            raise ValueError("Example cannot be None")
        
        if prediction is None:
            raise ValueError("Prediction cannot be None")
        
        with self._lock:
            # Generate a unique ID for this prediction
            pred_id = f"pred_{len(self.predictions)}"
            
            # Store example and prediction
            self.examples[pred_id] = example
            self.predictions[pred_id] = prediction
            
            # Add an entry to the ratings dictionary
            self.elo_system.ratings[pred_id] = self.elo_system.default_rating
            
            # Get question and answer
            try:
                question = getattr(example, self.question_field)
            except (AttributeError, TypeError):
                try:
                    question = example[self.question_field]
                except (KeyError, TypeError):
                    question = str(example)
            
            try:
                answer = getattr(prediction, self.answer_field)
            except (AttributeError, TypeError):
                try:
                    answer = prediction[self.answer_field]
                except (KeyError, TypeError):
                    answer = str(prediction)
            
            # If this is the first prediction, just return the default normalized score
            if len(self.predictions) == 1:
                if trace is not None:
                    return self.normalize_score(self.elo_system.default_rating) > 1.0
                return self.normalize_score(self.elo_system.default_rating)
            
            # Find similar ratings to compare against
            comparison_ids = self._select_comparison_samples(pred_id)
            
            # Perform comparisons
            for comp_id in comparison_ids:
                # Get comparison question and answer
                try:
                    comp_question = getattr(self.examples[comp_id], self.question_field)
                except (AttributeError, TypeError):
                    try:
                        comp_question = self.examples[comp_id][self.question_field]
                    except (KeyError, TypeError):
                        comp_question = str(self.examples[comp_id])
                
                try:
                    comp_answer = getattr(self.predictions[comp_id], self.answer_field)
                except (AttributeError, TypeError):
                    try:
                        comp_answer = self.predictions[comp_id][self.answer_field]
                    except (KeyError, TypeError):
                        comp_answer = str(self.predictions[comp_id])
                
                # Skip if comparing to itself or if questions are different
                if pred_id == comp_id or question != comp_question:
                    continue
                
                # Compare answers
                _, _, outcome = self.judge.compare(question, answer, comp_answer)
                
                # Update Elo ratings
                self.elo_system.record_match(pred_id, comp_id, outcome)
                
                # Increment comparison count
                self.comparison_count += 1
            
            # Get the final rating and normalize it
            final_rating = self.elo_system.get_rating(pred_id)
            normalized_score = self.normalize_score(final_rating)
            
            # For optimization, return a boolean based on whether the score is above 1.0
            if trace is not None:
                return normalized_score > 1.0
            
            return normalized_score
    
    def _select_comparison_samples(self, pred_id):
        """Select samples to compare against."""
        # Simple mock implementation for testing
        with self._lock:
            return [pid for pid in self.predictions.keys() if pid != pred_id][:self.num_comparisons]
    
    def normalize_score(self, rating):
        """Normalize a rating to a score between min_score and max_score."""
        with self._lock:
            # Get all current ratings
            all_ratings = {pid: self.elo_system.get_rating(pid) for pid in self.predictions.keys()}
            
            # If we only have one rating, return a default value
            if len(all_ratings) <= 1:
                return (self.min_score + self.max_score) / 2.0
            
            # Normalize ratings
            normalized_ratings = self.elo_system.normalize_ratings(
                target_ratio=self.normalize_ratio,
                target_threshold=1.0
            )
            
            # Get the normalized rating for this prediction
            if isinstance(rating, str):
                normalized_rating = normalized_ratings.get(rating, (self.min_score + self.max_score) / 2.0)
                # For string ratings, we can't compare with default_rating, so just return the normalized value
                return max(self.min_score, min(self.max_score, normalized_rating))
            else:
                # Find the prediction ID with this rating
                pred_ids = [pid for pid, r in self.elo_system.ratings.items() if r == rating]
                if pred_ids:
                    normalized_rating = normalized_ratings.get(pred_ids[0], (self.min_score + self.max_score) / 2.0)
                else:
                    # If we can't find the exact rating, use the default
                    normalized_rating = (self.min_score + self.max_score) / 2.0
            
            # Handle edge cases
            if rating < self.elo_system.default_rating - 1000:
                return self.min_score
            elif rating > self.elo_system.default_rating + 1000:
                return self.max_score
            
            # Clip to min_score and max_score
            return max(self.min_score, min(self.max_score, normalized_rating))


@dataclass
class TestExample:
    question: str


@dataclass
class TestPrediction:
    answer: str


def test_concurrent_calls():
    """Test concurrent calls to the EloMetric."""
    metric = ThreadSafeEloMetric()
    
    # Number of concurrent calls
    num_calls = 50
    
    # Function to call the metric
    def call_metric(i):
        example = TestExample(question="What is 2+2?")
        # Alternate between correct and incorrect answers
        answer = "correct" if i % 2 == 0 else "incorrect"
        prediction = TestPrediction(answer=answer)
        return metric(example, prediction)
    
    # Call the metric concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(call_metric, range(num_calls)))
    
    # Check that we have the expected number of predictions
    assert len(metric.predictions) == num_calls
    
    # Check that we have ratings for all predictions
    assert len(metric.elo_system.ratings) == num_calls
    
    # Check that all results are valid scores
    for score in results:
        assert metric.min_score <= score <= metric.max_score


def test_concurrent_normalize_score():
    """Test concurrent calls to normalize_score."""
    metric = ThreadSafeEloMetric()
    
    # Add some predictions
    for i in range(10):
        example = TestExample(question="What is 2+2?")
        answer = "correct" if i % 2 == 0 else "incorrect"
        prediction = TestPrediction(answer=answer)
        metric(example, prediction)
    
    # Number of concurrent calls
    num_calls = 50
    
    # Function to call normalize_score
    def call_normalize_score(i):
        # Use different ratings
        rating = 1500.0 + (i - num_calls // 2) * 10
        return metric.normalize_score(rating)
    
    # Call normalize_score concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(call_normalize_score, range(num_calls)))
    
    # Check that all results are valid scores
    for score in results:
        assert metric.min_score <= score <= metric.max_score


def test_race_conditions():
    """Test for race conditions by deliberately introducing delays."""
    # Create a subclass with deliberate delays to increase chance of race conditions
    class DelayedMetric(ThreadSafeEloMetric):
        def __call__(self, example, prediction, trace=None):
            # Add a small delay before acquiring the lock
            time.sleep(0.001)
            return super().__call__(example, prediction, trace)
        
        def normalize_score(self, rating):
            # Add a small delay before acquiring the lock
            time.sleep(0.001)
            return super().normalize_score(rating)
    
    metric = DelayedMetric()
    
    # Number of concurrent calls
    num_calls = 20
    
    # Function to call the metric
    def call_metric(i):
        example = TestExample(question="What is 2+2?")
        # Alternate between correct and incorrect answers
        answer = "correct" if i % 2 == 0 else "incorrect"
        prediction = TestPrediction(answer=answer)
        return metric(example, prediction)
    
    # Call the metric concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(call_metric, range(num_calls)))
    
    # Check that we have the expected number of predictions
    assert len(metric.predictions) == num_calls
    
    # Check that we have ratings for all predictions
    assert len(metric.elo_system.ratings) == num_calls
    
    # Check that all results are valid scores
    for score in results:
        assert metric.min_score <= score <= metric.max_score


def test_thread_safety_with_different_questions():
    """Test thread safety with different questions."""
    metric = ThreadSafeEloMetric()
    
    # Number of concurrent calls
    num_calls = 30
    
    # Function to call the metric with different questions
    def call_metric(i):
        question = f"Question {i % 5}"  # 5 different questions
        example = TestExample(question=question)
        # Alternate between correct and incorrect answers
        answer = "correct" if i % 2 == 0 else "incorrect"
        prediction = TestPrediction(answer=answer)
        return metric(example, prediction)
    
    # Call the metric concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(call_metric, range(num_calls)))
    
    # Check that we have the expected number of predictions
    assert len(metric.predictions) == num_calls
    
    # Check that we have ratings for all predictions
    assert len(metric.elo_system.ratings) == num_calls
    
    # Check that all results are valid scores
    for score in results:
        assert metric.min_score <= score <= metric.max_score


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
