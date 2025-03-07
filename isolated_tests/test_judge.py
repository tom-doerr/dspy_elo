"""
Isolated tests for the DSPy-based judge.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import from fully_isolated_test_elo_metric_direct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock version of the AnswerJudge class
class MockAnswerJudge:
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


@pytest.fixture
def mock_dspy_predict():
    """Mock the dspy.Predict class."""
    mock_instance = MagicMock()
    return mock_instance


def test_answer_judge_init():
    """Test AnswerJudge initialization."""
    criteria = "Test criteria"
    judge = MockAnswerJudge(criteria=criteria)
    
    assert judge.criteria == criteria


def test_compare_winner_a(mock_dspy_predict=None):
    """Test compare method when A wins."""
    # Create the judge
    judge = MockAnswerJudge(criteria="Test criteria")
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer AAAA",  # Longer answer
        answer_b="Answer B"
    )
    
    # Check the results
    assert winner == "A"
    assert explanation == "Answer A is more detailed"
    assert outcome == 1.0


def test_compare_winner_b(mock_dspy_predict=None):
    """Test compare method when B wins."""
    # Create the judge
    judge = MockAnswerJudge(criteria="Test criteria")
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer BBBB"  # Longer answer
    )
    
    # Check the results
    assert winner == "B"
    assert explanation == "Answer B is more detailed"
    assert outcome == 0.0


def test_compare_tie(mock_dspy_predict=None):
    """Test compare method when it's a tie."""
    # Create the judge
    judge = MockAnswerJudge(criteria="Test criteria")
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer B"  # Same length
    )
    
    # Check the results
    assert winner == "TIE"
    assert explanation == "Both answers are equally detailed"
    assert outcome == 0.5
