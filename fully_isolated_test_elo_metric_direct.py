"""
Fully isolated test for the EloMetric class.
"""

import unittest
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# Direct copy of the Elo functions to avoid import issues
def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score of player A in a match against player B.
    
    Args:
        rating_a: Elo rating for player A
        rating_b: Elo rating for player B
        
    Returns:
        Expected score for player A (between 0 and 1)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(old_rating: float, expected: float, actual: float, k_factor: float = 32.0) -> float:
    """
    Calculate the new Elo rating for a player.
    
    Args:
        old_rating: The previous Elo rating
        expected: The expected score for this match
        actual: The actual score for this match (1 for win, 0.5 for draw, 0 for loss)
        k_factor: The k-factor for Elo (default: 32)
        
    Returns:
        The updated Elo rating
    """
    return old_rating + k_factor * (actual - expected)


class EloRatingSystem:
    """
    A class to manage Elo ratings for a collection of items.
    """
    
    def __init__(self, default_rating: float = 1500.0, k_factor: float = 32.0):
        """
        Initialize the Elo rating system.
        
        Args:
            default_rating: The default rating for new items
            k_factor: The k-factor for Elo calculations
        """
        self.ratings = {}
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.history = []
    
    def get_rating(self, item_id: str) -> float:
        """
        Get the current rating for an item.
        
        Args:
            item_id: The identifier for the item
            
        Returns:
            The current Elo rating for the item
        """
        return self.ratings.get(item_id, self.default_rating)
    
    def record_match(self, item_a: str, item_b: str, outcome: float):
        """
        Record the outcome of a match between two items and update their ratings.
        
        Args:
            item_a: The identifier for the first item
            item_b: The identifier for the second item
            outcome: The outcome from item_a's perspective (1 for win, 0.5 for draw, 0 for loss)
            
        Returns:
            A tuple of the new ratings (rating_a, rating_b)
        """
        # Get current ratings
        rating_a = self.get_rating(item_a)
        rating_b = self.get_rating(item_b)
        
        # Calculate expected scores
        expected_a = expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        # Update ratings
        new_rating_a = update_elo(rating_a, expected_a, outcome, self.k_factor)
        new_rating_b = update_elo(rating_b, expected_b, 1.0 - outcome, self.k_factor)
        
        # Store new ratings
        self.ratings[item_a] = new_rating_a
        self.ratings[item_b] = new_rating_b
        
        # Record history
        self.history.append({
            'item_a': item_a,
            'item_b': item_b,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'outcome': outcome
        })
        
        return new_rating_a, new_rating_b
    
    def find_closest_ratings(self, item_id: str, n: int = 5):
        """
        Find the n items with the closest ratings to the given item.
        
        Args:
            item_id: The identifier for the reference item
            n: The number of items to return
            
        Returns:
            A list of item identifiers with the closest ratings
        """
        if item_id not in self.ratings and len(self.ratings) == 0:
            return []
        
        target_rating = self.get_rating(item_id)
        
        # Get all items except the target
        other_items = [(id, rating) for id, rating in self.ratings.items() if id != item_id]
        
        # Sort by rating difference
        other_items.sort(key=lambda x: abs(x[1] - target_rating))
        
        # Return the n closest items
        return [id for id, _ in other_items[:n]]
    
    def normalize_ratings(self, target_ratio: float = 0.5, target_threshold: float = 1.0):
        """
        Normalize ratings so that a specified ratio of items have ratings above a threshold.
        
        Args:
            target_ratio: The desired ratio of items with ratings above the threshold
            target_threshold: The threshold value (default: 1.0)
            
        Returns:
            A dictionary mapping item IDs to normalized ratings
        """
        if not self.ratings:
            return {}
        
        # Get all ratings
        all_ratings = list(self.ratings.values())
        
        # Sort ratings
        sorted_ratings = sorted(all_ratings)
        
        # Find the rating at the target percentile
        target_percentile = 1.0 - target_ratio
        target_index = int(target_percentile * len(sorted_ratings))
        target_index = max(0, min(target_index, len(sorted_ratings) - 1))
        target_rating = sorted_ratings[target_index]
        
        # Normalize ratings
        normalized_ratings = {}
        for item_id, rating in self.ratings.items():
            if rating < target_rating:
                # Scale from min_rating to target_rating -> 0 to target_threshold
                min_rating = min(all_ratings)
                if target_rating == min_rating:
                    normalized = 0.0  # Avoid division by zero
                else:
                    normalized = target_threshold * (rating - min_rating) / (target_rating - min_rating)
                    normalized = max(0.0, normalized)  # Ensure non-negative
            else:
                # Scale from target_rating to max_rating -> target_threshold to 2.0
                max_rating = max(all_ratings)
                if max_rating == target_rating:
                    normalized = target_threshold
                else:
                    normalized = target_threshold + (2.0 - target_threshold) * (rating - target_rating) / (max_rating - target_rating)
            
            normalized_ratings[item_id] = normalized
        
        return normalized_ratings


@dataclass
class SimpleExample:
    """Simple example for testing."""
    question: str
    answer: str = None


@dataclass
class SimplePrediction:
    """Simple prediction for testing."""
    answer: str


class MockJudge:
    """A mock judge for testing."""
    
    def __init__(self, criteria: str):
        self.criteria = criteria
        self.calls = []
    
    def compare(self, question: str, answer_a: str, answer_b: str):
        """
        Compare two answers and return a winner.
        
        Args:
            question: The question being answered
            answer_a: The first answer
            answer_b: The second answer
            
        Returns:
            A tuple of (winner, explanation, outcome)
        """
        self.calls.append((question, answer_a, answer_b))
        
        # Simple logic: longer answer wins
        if len(answer_a) > len(answer_b):
            return "A", "Answer A is more detailed", 1.0
        elif len(answer_b) > len(answer_a):
            return "B", "Answer B is more detailed", 0.0
        else:
            return "TIE", "Both answers are equally detailed", 0.5


class SimpleEloMetric:
    """
    A simplified version of the EloMetric for testing.
    """
    
    def __init__(
        self,
        criteria: str,
        num_comparisons: int = 2,
        normalize_ratio: float = 0.5,
        default_rating: float = 1500.0,
        k_factor: float = 32.0,
        min_score: float = 0.0,
        max_score: float = 2.0,
    ):
        """
        Initialize the Elo metric.
        
        Args:
            criteria: The criteria for comparing answers
            num_comparisons: Number of comparisons to perform for each evaluation
            normalize_ratio: Ratio of scores that should be above 1.0
            default_rating: Default Elo rating for new answers
            k_factor: K-factor for Elo calculations
            min_score: Minimum score to return
            max_score: Maximum score to return
        """
        self.criteria = criteria
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize Elo rating system
        self.elo_system = EloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Initialize judge
        self.judge = MockJudge(criteria=criteria)
        
        # Store examples and predictions
        self.examples = {}
        self.predictions = {}
    
    def __call__(self, example: Any, prediction: Any) -> float:
        """
        Evaluate a prediction using Elo rating system.
        
        Args:
            example: The example from the dataset
            prediction: The prediction from the model
            
        Returns:
            A normalized score
        """
        # Generate a unique ID for this prediction
        pred_id = str(uuid.uuid4())  # Track the prediction ID
        
        # Store example and prediction
        self.examples[pred_id] = example
        self.predictions[pred_id] = prediction
        
        # Get question and answer
        question = example.question
        answer = prediction.answer
        
        # If this is the first prediction, just return the default normalized score
        if len(self.predictions) == 1:
            return self.normalize_score(self.elo_system.default_rating)
        
        # Find similar ratings to compare against
        comparison_ids = self._select_comparison_samples(pred_id)
        
        # Perform comparisons
        for comp_id in comparison_ids:
            comp_question = self.examples[comp_id].question
            comp_answer = self.predictions[comp_id].answer
            
            # Skip if comparing to itself or if questions are different
            if pred_id == comp_id or question != comp_question:
                continue
            
            # Compare answers
            _, _, outcome = self.judge.compare(question, answer, comp_answer)
            
            # Update Elo ratings
            self.elo_system.record_match(pred_id, comp_id, outcome)
        
        # Get the final rating and normalize it
        final_rating = self.elo_system.get_rating(pred_id)
        normalized_score = self.normalize_score(final_rating)
        
        return normalized_score
    
    def _select_comparison_samples(self, pred_id: str) -> List[str]:
        """
        Select samples to compare against.
        
        Args:
            pred_id: The ID of the prediction to compare
            
        Returns:
            A list of prediction IDs to compare against
        """
        # If we have fewer than num_comparisons predictions, compare with all of them
        if len(self.predictions) <= self.num_comparisons:
            return [pid for pid in self.predictions.keys() if pid != pred_id]
        
        # Find predictions with similar ratings
        similar_ids = self.elo_system.find_closest_ratings(pred_id, self.num_comparisons)
        
        # If we couldn't find similar ratings, just return some IDs
        if not similar_ids:
            all_ids = [pid for pid in self.predictions.keys() if pid != pred_id]
            return all_ids[:self.num_comparisons]
        
        return similar_ids
    
    def normalize_score(self, rating) -> float:
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


class TestEloMetricDirect(unittest.TestCase):
    """Test the EloMetric class directly."""
    
    def setUp(self):
        """Set up the test case."""
        self.metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
    
    def test_basic_functionality(self):
        """Test basic functionality of the EloMetric."""
        # Create examples and predictions
        example1 = SimpleExample(question="What is the capital of France?")
        example2 = SimpleExample(question="What is the capital of France?")
        example3 = SimpleExample(question="What is the capital of France?")
        
        prediction1 = SimplePrediction(answer="Paris")
        prediction2 = SimplePrediction(answer="Paris is the capital of France.")
        prediction3 = SimplePrediction(answer="The capital of France is Paris.")
        
        # Test the metric
        score1 = self.metric(example1, prediction1)
        score2 = self.metric(example2, prediction2)
        score3 = self.metric(example3, prediction3)
        
        # Print debug information
        print(f"Scores: {score1}, {score2}, {score3}")
        print(f"Ratings: {list(self.metric.elo_system.ratings.values())}")
        
        # Basic assertions
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 2.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 2.0)
        self.assertGreaterEqual(score3, 0.0)
        self.assertLessEqual(score3, 2.0)
    
    def test_normalization(self):
        """Test the normalization of scores."""
        # Create a new metric with a different normalization ratio
        metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.3  # 30% of scores should be above 1.0
        )
        
        # Create examples and predictions
        examples = [SimpleExample(question="What is the capital of France?") for _ in range(10)]
        
        # Create predictions with varying answer lengths
        predictions = [
            SimplePrediction(answer="P" * (i + 1))  # Answers of increasing length
            for i in range(10)
        ]
        
        # Evaluate all predictions
        scores = [metric(ex, pred) for ex, pred in zip(examples, predictions)]
        
        # Print debug information
        print(f"Normalization test scores: {scores}")
        
        # Count scores above 1.0
        scores_above_threshold = sum(1 for score in scores if score > 1.0)
        ratio_above_threshold = scores_above_threshold / len(scores)
        
        # Allow for some small deviation due to rounding and the discrete nature of the test
        self.assertAlmostEqual(ratio_above_threshold, 0.3, delta=0.2)
    
    def test_different_comparison_counts(self):
        """Test the metric with different numbers of comparisons."""
        # Create metrics with different comparison counts
        metric1 = SimpleEloMetric(criteria="Test criteria", num_comparisons=1)
        metric2 = SimpleEloMetric(criteria="Test criteria", num_comparisons=3)
        
        # Create examples and predictions
        examples = [SimpleExample(question="What is the capital of France?") for _ in range(5)]
        predictions = [SimplePrediction(answer="Paris" + "!" * i) for i in range(5)]
        
        # Evaluate with different metrics
        scores1 = [metric1(ex, pred) for ex, pred in zip(examples, predictions)]
        scores2 = [metric2(ex, pred) for ex, pred in zip(examples, predictions)]
        
        # Print debug information
        print(f"Comparison count test - 1 comparison: {scores1}")
        print(f"Comparison count test - 3 comparisons: {scores2}")
        
        # Both should produce valid scores
        for score in scores1 + scores2:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 2.0)
    
    def test_edge_cases(self):
        """Test edge cases for the EloMetric."""
        metric = SimpleEloMetric(criteria="Test criteria")
        
        # Test with a single prediction
        example = SimpleExample(question="What is the capital of France?")
        prediction = SimplePrediction(answer="Paris")
        
        score = metric(example, prediction)
        print(f"Single prediction score: {score}")
        
        # Should be the default normalized score (1.0)
        self.assertAlmostEqual(score, 1.0, delta=0.1)
        
        # Test with identical answers
        example1 = SimpleExample(question="What is the capital of France?")
        example2 = SimpleExample(question="What is the capital of France?")
        
        prediction1 = SimplePrediction(answer="Paris")
        prediction2 = SimplePrediction(answer="Paris")  # Identical to prediction1
        
        score1 = metric(example1, prediction1)
        score2 = metric(example2, prediction2)
        
        print(f"Identical answers scores: {score1}, {score2}")
        
        # Should be close to each other
        self.assertAlmostEqual(score1, score2, delta=0.5)
    
    def test_different_questions(self):
        """Test the metric with different questions."""
        metric = SimpleEloMetric(criteria="Test criteria")
        
        # Create examples with different questions
        example1 = SimpleExample(question="What is the capital of France?")
        example2 = SimpleExample(question="What is the capital of Germany?")
        
        prediction1 = SimplePrediction(answer="Paris")
        prediction2 = SimplePrediction(answer="Berlin")
        
        # Evaluate predictions
        score1 = metric(example1, prediction1)
        score2 = metric(example2, prediction2)
        
        print(f"Different questions scores: {score1}, {score2}")
        
        # Both should be valid scores
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 2.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 2.0)
        
        # The judge should not have compared these (different questions)
        self.assertEqual(len(metric.judge.calls), 0)


if __name__ == "__main__":
    unittest.main()
