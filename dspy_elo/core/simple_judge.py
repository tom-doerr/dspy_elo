"""
Simple judge implementation without DSPy dependencies.
"""

from typing import Tuple, Any, Callable, Optional


class SimpleJudge:
    """
    A simple judge implementation that doesn't depend on DSPy.
    This can be used for simple comparisons or as a placeholder for testing.
    """
    
    def __init__(
        self,
        compare_fn: Optional[Callable[[str, Any, Any], Tuple[str, str, float]]] = None,
        criteria: str = "accuracy"
    ):
        """
        Initialize a SimpleJudge.
        
        Args:
            compare_fn: Optional function to use for comparing answers
            criteria: Criteria to use for comparison (only used if compare_fn is not provided)
        """
        self.compare_fn = compare_fn
        self.criteria = criteria
    
    def compare(self, question: str, answer_a: Any, answer_b: Any) -> Tuple[str, str, float]:
        """
        Compare two answers and return the winner, explanation, and outcome.
        
        Args:
            question: Question being answered
            answer_a: First answer
            answer_b: Second answer
            
        Returns:
            Tuple of (winner, explanation, outcome)
            winner: "A", "B", or "TIE"
            explanation: Explanation of the decision
            outcome: 1.0 for A wins, 0.0 for B wins, 0.5 for tie
        """
        # If a compare function was provided, use it
        if self.compare_fn:
            return self.compare_fn(question, answer_a, answer_b)
        
        # Otherwise, use a simple length-based comparison
        # This is just a placeholder - in a real implementation, you would
        # use a more sophisticated comparison method
        len_a = len(str(answer_a))
        len_b = len(str(answer_b))
        
        if len_a > len_b:
            return "A", f"A is longer ({len_a} vs {len_b})", 1.0
        elif len_b > len_a:
            return "B", f"B is longer ({len_b} vs {len_a})", 0.0
        else:
            return "TIE", "Both answers have the same length", 0.5
