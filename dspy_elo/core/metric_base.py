"""
Base implementation of the Elo metric without DSPy dependencies.
"""

import threading
import uuid
from typing import Dict, Any, List, Union, Optional, Callable, Tuple

from .elo_rating import EloRatingSystem


class BaseEloMetric:
    """
    Base class for Elo-based metrics without DSPy dependencies.
    This class provides the core functionality for Elo-based metrics,
    but requires a judge implementation to be provided.
    """
    
    def __init__(
        self,
        judge_fn: Callable[[str, Any, Any], Tuple[str, str, float]],
        num_comparisons: int = 5,
        normalize_ratio: float = 0.5,
        default_rating: float = 1500.0,
        k_factor: float = 32.0,
        min_score: float = 0.0,
        max_score: float = 2.0,
        question_field: str = "question",
        answer_field: str = "answer",
    ):
        """
        Initialize a BaseEloMetric.
        
        Args:
            judge_fn: Function that compares two answers and returns (winner, explanation, outcome)
            num_comparisons: Number of comparisons to perform per evaluation
            normalize_ratio: Ratio of scores that should be above 1.0
            default_rating: Default Elo rating for new items
            k_factor: K-factor for Elo calculation
            min_score: Minimum normalized score
            max_score: Maximum normalized score
            question_field: Field name for the question in examples
            answer_field: Field name for the answer in predictions
        """
        # Validate parameters
        if num_comparisons < 0:
            raise ValueError("num_comparisons must be non-negative")
        
        if normalize_ratio < 0 or normalize_ratio > 1:
            raise ValueError("normalize_ratio must be between 0 and 1")
        
        if min_score > max_score:
            raise ValueError("min_score must be less than or equal to max_score")
        
        self.judge_fn = judge_fn
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.min_score = min_score
        self.max_score = max_score
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Initialize Elo rating system
        self.elo_system = EloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Store examples and predictions
        self.examples = {}
        self.predictions = {}
        
        # Track number of comparisons performed
        self.comparison_count = 0
        
        # Add a lock for thread safety
        self._lock = threading.RLock()
    
    def __call__(self, example: Any, prediction: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        """
        Evaluate a prediction against an example using Elo ratings.
        
        Args:
            example: Example to evaluate against
            prediction: Prediction to evaluate
            trace: Optional trace information
            
        Returns:
            Normalized Elo score for the prediction, or a boolean if trace is provided
        """
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
            question = self._extract_field(example, self.question_field, str(example))
            answer = self._extract_field(prediction, self.answer_field, str(prediction))
            
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
                comp_question = self._extract_field(self.examples[comp_id], self.question_field, str(self.examples[comp_id]))
                comp_answer = self._extract_field(self.predictions[comp_id], self.answer_field, str(self.predictions[comp_id]))
                
                # Skip if comparing to itself or if questions are different
                if pred_id == comp_id or question != comp_question:
                    continue
                
                # Compare answers
                _, _, outcome = self.judge_fn(question, answer, comp_answer)
                
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
    
    def _extract_field(self, obj: Any, field: str, default: Any) -> Any:
        """
        Extract a field from an object, handling different object types.
        
        Args:
            obj: Object to extract field from
            field: Field name to extract
            default: Default value if field is not found
            
        Returns:
            Value of the field, or default if not found
        """
        try:
            return getattr(obj, field)
        except (AttributeError, TypeError):
            try:
                return obj[field]
            except (KeyError, TypeError):
                return default
    
    def _select_comparison_samples(self, pred_id: str) -> List[str]:
        """
        Select samples to compare against.
        
        Args:
            pred_id: ID of the prediction to compare
            
        Returns:
            List of prediction IDs to compare against
        """
        with self._lock:
            # If we have fewer than num_comparisons predictions, compare against all of them
            if len(self.predictions) <= self.num_comparisons + 1:
                return [pid for pid in self.predictions.keys() if pid != pred_id]
            
            # Otherwise, find the closest ratings
            return self.elo_system.find_closest_ratings(pred_id, self.num_comparisons)
    
    def normalize_score(self, rating: Union[str, float]) -> float:
        """
        Normalize a rating to a score between min_score and max_score.
        
        Args:
            rating: Rating to normalize, either a prediction ID or a raw rating
            
        Returns:
            Normalized score
        """
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
