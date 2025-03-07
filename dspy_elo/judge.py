"""
DSPy-based judge for comparing answers based on custom criteria.
"""

import dspy
from typing import Dict, List, Optional, Tuple, Union, Any


class CompareAnswers(dspy.Signature):
    """
    Signature for comparing two answers based on custom criteria.
    """
    question = dspy.InputField(desc="The question being answered")
    answer_a = dspy.InputField(desc="The first answer to compare")
    answer_b = dspy.InputField(desc="The second answer to compare")
    criteria = dspy.InputField(desc="The criteria for comparing the answers")
    
    winner = dspy.OutputField(desc="Which answer is better: 'A', 'B', or 'TIE'")
    explanation = dspy.OutputField(desc="Explanation of why the chosen answer is better")


class AnswerJudge:
    """
    A DSPy-based judge for comparing answers based on custom criteria.
    """
    
    def __init__(self, criteria: str):
        """
        Initialize the judge with custom criteria.
        
        Args:
            criteria: The criteria for comparing answers
        """
        self.criteria = criteria
        self.predictor = dspy.Predict(CompareAnswers)
    
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
        result = self.predictor(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
            criteria=self.criteria
        )
        
        winner = result.winner.strip().upper()
        explanation = result.explanation
        
        # Convert winner to outcome score for answer_a
        if winner == 'A':
            outcome = 1.0
        elif winner == 'B':
            outcome = 0.0
        else:  # TIE
            outcome = 0.5
            
        return winner, explanation, outcome


class OptimizedAnswerJudge(AnswerJudge):
    """
    An optimized version of the AnswerJudge that can be compiled with DSPy.
    """
    
    def __init__(self, criteria: str, compiled_module: Optional[dspy.Module] = None):
        """
        Initialize the optimized judge.
        
        Args:
            criteria: The criteria for comparing answers
            compiled_module: A pre-compiled DSPy module
        """
        super().__init__(criteria)
        
        if compiled_module is not None:
            self.predictor = compiled_module
    
    def compile(self, examples: List[Dict[str, Any]], optimizer: dspy.Optimizer) -> 'OptimizedAnswerJudge':
        """
        Compile the judge using DSPy optimization.
        
        Args:
            examples: List of examples for optimization
            optimizer: DSPy optimizer to use
            
        Returns:
            An optimized version of the judge
        """
        compiled_module = optimizer.compile(
            self.predictor,
            examples=examples
        )
        
        return OptimizedAnswerJudge(self.criteria, compiled_module)
