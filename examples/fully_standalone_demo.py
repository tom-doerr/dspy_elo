"""
Fully standalone demonstration of the Elo rating system.
This file contains all the necessary code without importing from the dspy_elo package.
"""

import math
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


class EloRatingSystem:
    """
    A class for managing Elo ratings for a set of items.
    """
    
    def __init__(self, default_rating: float = 1500.0, k_factor: float = 32.0):
        """
        Initialize the Elo rating system.
        
        Args:
            default_rating: The default rating for new items
            k_factor: The K-factor for Elo calculations
        """
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.history = []
    
    def get_rating(self, item_id: str) -> float:
        """
        Get the rating for an item.
        
        Args:
            item_id: The ID of the item
            
        Returns:
            The rating for the item
        """
        return self.ratings.get(item_id, self.default_rating)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score for item A when playing against item B.
        
        Args:
            rating_a: The rating of item A
            rating_b: The rating of item B
            
        Returns:
            The expected score for item A (between 0 and 1)
        """
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
    
    def update_rating(self, rating: float, expected_score: float, actual_score: float) -> float:
        """
        Update a rating based on the expected and actual scores.
        
        Args:
            rating: The current rating
            expected_score: The expected score
            actual_score: The actual score
            
        Returns:
            The updated rating
        """
        return rating + self.k_factor * (actual_score - expected_score)
    
    def record_match(self, item_a: str, item_b: str, outcome: float) -> None:
        """
        Record a match between two items and update their ratings.
        
        Args:
            item_a: The ID of the first item
            item_b: The ID of the second item
            outcome: The outcome of the match from item_a's perspective
                     (1.0 for win, 0.5 for draw, 0.0 for loss)
        """
        # Get current ratings
        rating_a = self.get_rating(item_a)
        rating_b = self.get_rating(item_b)
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        # Update ratings
        new_rating_a = self.update_rating(rating_a, expected_a, outcome)
        new_rating_b = self.update_rating(rating_b, expected_b, 1.0 - outcome)
        
        # Store new ratings
        self.ratings[item_a] = new_rating_a
        self.ratings[item_b] = new_rating_b
        
        # Record match in history
        self.history.append({
            'item_a': item_a,
            'item_b': item_b,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'expected_a': expected_a,
            'expected_b': expected_b,
            'outcome': outcome
        })
    
    def find_closest_ratings(self, item_id: str, n: int = 1) -> List[str]:
        """
        Find the n items with the closest ratings to the given item.
        
        Args:
            item_id: The ID of the item to find similar ratings for
            n: The number of similar items to return
            
        Returns:
            A list of item IDs with similar ratings
        """
        if item_id not in self.ratings and len(self.ratings) == 0:
            return []
        
        target_rating = self.get_rating(item_id)
        
        # Sort items by rating difference
        sorted_items = sorted(
            [i for i in self.ratings.keys() if i != item_id],
            key=lambda i: abs(self.ratings[i] - target_rating)
        )
        
        # Return the n closest items
        return sorted_items[:n]
    
    def normalize_ratings(
        self,
        target_ratio: float = 0.5,
        target_threshold: float = 1.0
    ) -> Dict[str, float]:
        """
        Normalize ratings so that a certain ratio of them are above a threshold.
        
        Args:
            target_ratio: The ratio of ratings that should be above the threshold
            target_threshold: The threshold value
            
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
            # Linear scaling: target_rating -> target_threshold
            if rating <= target_rating:
                # Scale from min_rating to target_rating -> 0 to target_threshold
                min_rating = min(all_ratings)
                if target_rating == min_rating:
                    normalized = target_threshold * (rating / target_rating)
                else:
                    normalized = target_threshold * (rating - min_rating) / (target_rating - min_rating)
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
class QAExample:
    question: str
    answer: str = None


@dataclass
class Prediction:
    answer: str


class MockJudge:
    """A mock judge that compares answers based on length."""
    
    def __init__(self, criteria: str):
        self.criteria = criteria
    
    def compare(self, question: str, answer_a: str, answer_b: str) -> Tuple[str, str, float]:
        """
        Compare two answers and determine which is better.
        
        Args:
            question: The question being answered
            answer_a: The first answer to compare
            answer_b: The second answer to compare
            
        Returns:
            A tuple containing:
            - The winner ('A', 'B', or 'TIE')
            - The explanation for the decision
            - The outcome score for answer_a (1.0 for win, 0.5 for tie, 0.0 for loss)
        """
        # Simple logic: longer answer wins
        if len(answer_a) > len(answer_b):
            return "A", "A is more detailed", 1.0
        elif len(answer_b) > len(answer_a):
            return "B", "B is more detailed", 0.0
        else:
            return "TIE", "Both are equally detailed", 0.5


class SimpleEloMetric:
    """
    A simplified version of the EloMetric for demonstration purposes.
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
    
    def normalize_score(self, rating: float) -> float:
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


def main():
    print("Standalone Elo Rating System Demonstration")
    print("------------------------------------------")
    
    # Create an Elo metric
    criteria = """
    Compare these two answers based on:
    1. Accuracy - Which answer is more factually correct?
    2. Completeness - Which answer addresses more aspects of the question?
    3. Clarity - Which answer is more clearly written and easier to understand?
    
    Choose the better answer based on these criteria.
    """
    
    elo_metric = SimpleEloMetric(
        criteria=criteria,
        num_comparisons=2,
        normalize_ratio=0.5
    )
    
    # Create some examples and predictions
    examples = [
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
    ]
    
    predictions = [
        Prediction(answer="Paris is the capital of France."),
        Prediction(answer="The capital of France is Paris, a city known for its art, culture, and the Eiffel Tower."),
        Prediction(answer="Paris."),
        Prediction(answer="The capital city of the French Republic is Paris, located in the north-central part of the country on the Seine River. It is one of the world's most important cultural, financial, and commercial centers."),
        Prediction(answer="France's capital is Paris.")
    ]
    
    # Evaluate each example
    print("\nEvaluating predictions...")
    scores = []
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"\nPrediction {i+1}:")
        print(f"Question: {example.question}")
        print(f"Answer: {prediction.answer}")
        
        score = elo_metric(example, prediction)
        scores.append(score)
        print(f"Score: {score:.4f}")
    
    print("\nFinal Elo ratings:")
    for i, (pred_id, rating) in enumerate(elo_metric.elo_system.ratings.items()):
        normalized_score = elo_metric.normalize_score(rating)
        answer = elo_metric.predictions[pred_id].answer
        print(f"Answer {i+1}: {answer[:30]}..." if len(answer) > 30 else f"Answer {i+1}: {answer}")
        print(f"Raw Elo Rating: {rating:.2f}")
        print(f"Normalized Score: {normalized_score:.4f}")
        print("-" * 40)
    
    print(f"\nAverage score: {sum(scores) / len(scores):.4f}")
    
    # Show match history
    print("\nMatch history:")
    for i, match in enumerate(elo_metric.elo_system.history):
        item_a = match['item_a']
        item_b = match['item_b']
        answer_a = elo_metric.predictions[item_a].answer
        answer_b = elo_metric.predictions[item_b].answer
        
        print(f"\nMatch {i+1}:")
        print(f"Answer A: {answer_a[:30]}..." if len(answer_a) > 30 else f"Answer A: {answer_a}")
        print(f"Answer B: {answer_b[:30]}..." if len(answer_b) > 30 else f"Answer B: {answer_b}")
        print(f"Outcome: {'A wins' if match['outcome'] == 1.0 else 'B wins' if match['outcome'] == 0.0 else 'Tie'}")
        print(f"Rating change for A: {match['rating_a_before']:.2f} → {match['rating_a_after']:.2f}")
        print(f"Rating change for B: {match['rating_b_before']:.2f} → {match['rating_b_after']:.2f}")


if __name__ == "__main__":
    main()
