"""
Elo-based metric for DSPy.
"""

import dspy
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .elo import EloRatingSystem
from .judge import AnswerJudge, OptimizedAnswerJudge


class EloMetric:
    """
    An Elo-based metric for evaluating DSPy model outputs.
    """
    
    def __init__(
        self,
        criteria: str,
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
        Initialize the Elo metric.
        
        Args:
            criteria: The criteria for comparing answers
            num_comparisons: Number of comparisons to perform for each evaluation
            normalize_ratio: Ratio of scores that should be above 1.0
            default_rating: Default Elo rating for new answers
            k_factor: K-factor for Elo calculations
            min_score: Minimum score to return
            max_score: Maximum score to return
            question_field: Field name for the question in examples
            answer_field: Field name for the answer in predictions
        """
        self.criteria = criteria
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.min_score = min_score
        self.max_score = max_score
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Initialize Elo rating system
        self.elo_system = EloRatingSystem(default_rating=default_rating, k_factor=k_factor)
        
        # Initialize judge
        self.judge = AnswerJudge(criteria=criteria)
        
        # Store examples and predictions
        self.examples: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
    
    def __call__(self, example: Any, prediction: Any, trace: Optional[List] = None) -> float:
        """
        Evaluate a prediction using Elo rating system.
        
        Args:
            example: The example from the dataset
            prediction: The prediction from the model
            trace: Optional trace for optimization
            
        Returns:
            A score between min_score and max_score
        """
        # Generate a unique ID for this prediction
        pred_id = str(uuid.uuid4())  # Track the prediction ID
        
        # Store example and prediction
        self.examples[pred_id] = example
        self.predictions[pred_id] = prediction
        
        # Get question and answer
        try:
            question = getattr(example, self.question_field)
        except (AttributeError, TypeError):
            # Try dictionary-like access if attribute access fails
            try:
                question = example[self.question_field]
            except (KeyError, TypeError):
                # If all else fails, use the example itself as the question
                question = str(example)
        
        try:
            answer = getattr(prediction, self.answer_field)
        except (AttributeError, TypeError):
            # Try dictionary-like access if attribute access fails
            try:
                answer = prediction[self.answer_field]
            except (KeyError, TypeError):
                # If all else fails, use the prediction itself as the answer
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
            # Get comparison question and answer using the same logic as above
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
        
        # Get the final rating and normalize it
        final_rating = self.elo_system.get_rating(pred_id)
        normalized_score = self.normalize_score(final_rating)
        
        # For optimization, return a boolean based on whether the score is above 1.0
        if trace is not None:
            return normalized_score > 1.0
        
        return normalized_score
    
    def _select_comparison_samples(self, pred_id: str) -> List[str]:
        """
        Select samples to compare against based on similar Elo ratings.
        
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
        
        # If we couldn't find similar ratings, randomly select some
        if not similar_ids:
            all_ids = [pid for pid in self.predictions.keys() if pid != pred_id]
            # Use a deterministic selection if numpy is not available
            if not hasattr(np, 'random'):
                return all_ids[:min(self.num_comparisons, len(all_ids))]
            return np.random.choice(all_ids, min(self.num_comparisons, len(all_ids)), replace=False).tolist()
        
        return similar_ids
    
    def normalize_score(self, rating: float) -> float:
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
    
    def optimize(self, examples: List[Dict[str, Any]], optimizer: dspy.Optimizer) -> 'EloMetric':
        """
        Optimize the judge using DSPy optimization.
        
        Args:
            examples: List of examples for optimization
            optimizer: DSPy optimizer to use
            
        Returns:
            An optimized version of the metric
        """
        optimized_judge = OptimizedAnswerJudge(self.criteria)
        optimized_judge = optimized_judge.compile(examples, optimizer)
        
        # Create a new metric with the optimized judge
        optimized_metric = EloMetric(
            criteria=self.criteria,
            num_comparisons=self.num_comparisons,
            normalize_ratio=self.normalize_ratio,
            default_rating=self.elo_system.default_rating,
            k_factor=self.elo_system.k_factor,
            min_score=self.min_score,
            max_score=self.max_score,
            question_field=self.question_field,
            answer_field=self.answer_field,
        )
        
        optimized_metric.judge = optimized_judge
        
        return optimized_metric
