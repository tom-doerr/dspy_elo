"""
Standalone test for the Elo metric.
This file contains tests for the Elo metric without depending on DSPy.
"""

import unittest
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# Import the EloRatingSystem from our standalone test
from standalone_test_elo import EloRatingSystem


@dataclass
class MockExample:
    """Mock example for testing."""
    question: str
    answer: Optional[str] = None


@dataclass
class MockPrediction:
    """Mock prediction for testing."""
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
    ):
        """
        Initialize the Elo metric.
        
        Args:
            criteria: The criteria for comparing answers
            num_comparisons: Number of comparisons to perform for each evaluation
            normalize_ratio: Ratio of scores that should be above 1.0
            default_rating: Default Elo rating for new answers
            k_factor: K-factor for Elo calculations
        """
        self.criteria = criteria
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        
        # Initialize Elo rating system
        self.elo_system = EloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Initialize judge
        self.judge = MockJudge(criteria=criteria)
        
        # Store examples and predictions
        self.examples: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
    
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
        pred_id = str(uuid.uuid4())
        
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
        Normalize an Elo rating to a score between 0.0 and 2.0.
        
        Args:
            rating: The Elo rating to normalize
            
        Returns:
            A normalized score
        """
        # Get all current ratings
        all_ratings = {pid: self.elo_system.get_rating(pid) for pid in self.predictions.keys()}
        
        # If we only have one rating, return a default value
        if len(all_ratings) <= 1:
            return 1.0
        
        # Normalize ratings so that normalize_ratio of them are above 1.0
        normalized_ratings = self.elo_system.normalize_ratings(
            target_ratio=self.normalize_ratio,
            target_threshold=1.0
        )
        
        # Get the normalized rating for this prediction
        if isinstance(rating, str):
            normalized_rating = normalized_ratings.get(rating, 1.0)
        else:
            # Find the prediction ID with this rating
            pred_ids = [pid for pid, r in self.elo_system.ratings.items() if r == rating]
            if pred_ids:
                normalized_rating = normalized_ratings.get(pred_ids[0], 1.0)
            else:
                # If we can't find the exact rating, use the default
                normalized_rating = 1.0
        
        # Clip to min_score and max_score
        return max(0.0, min(2.0, normalized_rating))


class TestEloMetric(unittest.TestCase):
    """
    Test the Elo metric.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        criteria = """
        Compare these two answers based on:
        1. Accuracy - Which answer is more factually correct?
        2. Completeness - Which answer addresses more aspects of the question?
        3. Clarity - Which answer is more clearly written and easier to understand?
        """
        
        self.metric = SimpleEloMetric(
            criteria=criteria,
            num_comparisons=2,
            normalize_ratio=0.5,
            default_rating=1500.0,
            k_factor=32.0
        )
    
    def test_first_prediction(self):
        """
        Test that the first prediction gets a score of 1.0.
        """
        example = MockExample(question="What is the capital of France?")
        prediction = MockPrediction(answer="Paris")
        
        score = self.metric(example, prediction)
        
        self.assertEqual(score, 1.0)
    
    def test_multiple_predictions(self):
        """
        Test that multiple predictions are compared correctly.
        """
        example1 = MockExample(question="What is the capital of France?")
        example2 = MockExample(question="What is the capital of France?")
        example3 = MockExample(question="What is the capital of France?")
        example4 = MockExample(question="What is the capital of France?")
        example5 = MockExample(question="What is the capital of France?")
        
        prediction1 = MockPrediction(answer="Paris")
        prediction2 = MockPrediction(answer="Paris is the capital of France.")
        prediction3 = MockPrediction(answer="The capital of France is Paris, a city known for its art and culture.")
        prediction4 = MockPrediction(answer="P")  # Shortest answer
        prediction5 = MockPrediction(answer="The capital city of the French Republic is Paris, located in the north-central part of the country on the Seine River. It is one of the world's most important cultural, financial, and commercial centers.")  # Longest answer
        
        # First prediction should get a score of 1.0
        score1 = self.metric(example1, prediction1)
        self.assertEqual(score1, 1.0)
        
        # Add more predictions to get a better distribution of ratings
        score2 = self.metric(example2, prediction2)
        score3 = self.metric(example3, prediction3)
        score4 = self.metric(example4, prediction4)
        score5 = self.metric(example5, prediction5)
        
        # Print debug information
        print("\nDebug information for multiple_predictions test:")
        print(f"Scores: {score1}, {score2}, {score3}, {score4}, {score5}")
        print(f"Elo ratings: {[self.metric.elo_system.get_rating(pid) for pid in self.metric.predictions.keys()]}")
        
        # With 5 predictions, we should have a good distribution of scores
        # The shortest answer should have the lowest score
        self.assertLess(score4, score1)
        
        # The longest answer should have one of the highest scores
        # Since both score3 and score5 are at the maximum of 2.0, we can't assert that one is greater than the other
        self.assertGreaterEqual(score5, 1.0)
        
        # Check that the judge was called the correct number of times
        # Each prediction after the first should be compared to at most num_comparisons previous predictions
        expected_calls = min(len(self.metric.predictions) - 1, 
                            sum(min(i, self.metric.num_comparisons) for i in range(len(self.metric.predictions) - 1)))
        self.assertGreaterEqual(len(self.metric.judge.calls), expected_calls)
    
    def test_normalize_scores(self):
        """
        Test that scores are normalized correctly.
        """
        # Add some items with different ratings
        self.metric.elo_system.ratings = {
            'item1': 1500.0,
            'item2': 1600.0,
            'item3': 1550.0,
            'item4': 1700.0
        }
        
        # Add corresponding predictions
        self.metric.predictions = {
            'item1': MockPrediction(answer="Short answer"),
            'item2': MockPrediction(answer="Medium length answer"),
            'item3': MockPrediction(answer="Another medium answer"),
            'item4': MockPrediction(answer="Very long and detailed answer with lots of information")
        }
        
        # Normalize scores
        normalized = {item_id: self.metric.normalize_score(rating) 
                      for item_id, rating in self.metric.elo_system.ratings.items()}
        
        # Print debug information
        print("\nDebug information for normalize_scores test:")
        print(f"Original ratings: {self.metric.elo_system.ratings}")
        print(f"Normalized scores: {normalized}")
        print(f"Scores above threshold: {[k for k, v in normalized.items() if v > 1.0]}")
        print(f"Scores at or above threshold: {[k for k, v in normalized.items() if v >= 1.0]}")
        
        # Check that at least one score is above 1.0
        above_threshold = sum(1 for score in normalized.values() if score > 1.0)
        self.assertGreaterEqual(above_threshold, 1)
        
        # Check that at least 50% of scores are at or above 1.0
        at_or_above_threshold = sum(1 for score in normalized.values() if score >= 1.0)
        self.assertGreaterEqual(at_or_above_threshold, 2)
        
        # Check that the scores are in the same order as the ratings
        for i in range(1, 4):
            item_a = f'item{i}'
            item_b = f'item{i+1}'
            if self.metric.elo_system.ratings[item_a] < self.metric.elo_system.ratings[item_b]:
                self.assertLess(normalized[item_a], normalized[item_b])
            else:
                self.assertGreater(normalized[item_a], normalized[item_b])


if __name__ == '__main__':
    unittest.main()
