"""
Elo-based metric for DSPy with modular design to minimize DSPy dependencies.
"""

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

import uuid
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .core import BaseEloMetric
from .core.simple_judge import SimpleJudge

# Only import DSPy-dependent modules if DSPy is available
if DSPY_AVAILABLE:
    from .judge import AnswerJudge, OptimizedAnswerJudge


class EloMetric(BaseEloMetric):
    """
    An Elo-based metric for evaluating DSPy model outputs.
    This class extends BaseEloMetric to add DSPy-specific functionality.
    """
    
    def __init__(
        self,
        criteria: str = "accuracy",
        num_comparisons: int = 5,
        normalize_ratio: float = 0.5,
        default_rating: float = 1500.0,
        k_factor: float = 32.0,
        min_score: float = 0.0,
        max_score: float = 2.0,
        question_field: str = "question",
        answer_field: str = "answer",
        use_dspy_judge: bool = True,
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
            use_dspy_judge: Whether to use DSPy's AnswerJudge (if available)
        """
        # Check if DSPy is available when use_dspy_judge is True
        if use_dspy_judge and not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is not available. Install it with 'pip install dspy' or set use_dspy_judge=False."
            )
        
        # Create the judge
        if use_dspy_judge and DSPY_AVAILABLE:
            self.judge = AnswerJudge(criteria=criteria)
            judge_fn = self.judge.compare
        else:
            self.judge = SimpleJudge(criteria=criteria)
            judge_fn = self.judge.compare
        
        # Initialize the base class
        super().__init__(
            judge_fn=judge_fn,
            num_comparisons=num_comparisons,
            normalize_ratio=normalize_ratio,
            default_rating=default_rating,
            k_factor=k_factor,
            min_score=min_score,
            max_score=max_score,
            question_field=question_field,
            answer_field=answer_field,
        )
        
        # Store the criteria for potential optimization
        self.criteria = criteria
        self.use_dspy_judge = use_dspy_judge
    
    def optimize(self, examples: List[Dict[str, Any]], optimizer: Any) -> 'EloMetric':
        """
        Optimize the judge using DSPy optimization.
        
        Args:
            examples: List of examples for optimization
            optimizer: DSPy optimizer to use
            
        Returns:
            An optimized version of the metric
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not available. Install it with 'pip install dspy'.")
        
        if not self.use_dspy_judge:
            raise ValueError("Cannot optimize a metric that doesn't use DSPy's AnswerJudge.")
        
        # Create an optimized judge
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
            use_dspy_judge=True,
        )
        
        # Replace the judge
        optimized_metric.judge = optimized_judge
        optimized_metric.judge_fn = optimized_judge.compare
        
        return optimized_metric
