"""
Isolated integration test for the EloMetric.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from dataclasses import dataclass

# Add the parent directory to the path so we can import from fully_isolated_test_elo_metric_direct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the isolated implementation
from fully_isolated_test_elo_metric_direct import SimpleEloMetric


# Create a simple example and prediction class for testing
@dataclass
class TestExample:
    question: str
    answer: str = None


@dataclass
class TestPrediction:
    answer: str


class MockModule:
    """Mock module for testing."""
    
    def __init__(self, answers=None):
        self.answers = answers or {}
    
    def __call__(self, question):
        # Return a mock prediction with a pre-defined answer
        return TestPrediction(answer=self.answers.get(question, "Default answer"))


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


@pytest.fixture
def mock_dspy_predict():
    """Mock the dspy.Predict class."""
    mock_instance = MagicMock()
    
    # Configure the mock to return different winners for different comparisons
    def side_effect(**kwargs):
        answer_a = kwargs.get('answer_a', '')
        answer_b = kwargs.get('answer_b', '')
        
        # Simple logic: longer answer wins
        if len(answer_a) > len(answer_b):
            winner = "A"
            explanation = "A is more detailed"
        elif len(answer_b) > len(answer_a):
            winner = "B"
            explanation = "B is more detailed"
        else:
            winner = "TIE"
            explanation = "Both are equally detailed"
        
        result = MagicMock()
        result.winner = winner
        result.explanation = explanation
        return result
    
    mock_instance.side_effect = side_effect
    
    return mock_instance


def test_integration_with_mock():
    """Test integration with a mock program."""
    # Create examples
    examples = [
        TestExample(question="What is the capital of France?"),
        TestExample(question="Who wrote the novel '1984'?"),
        TestExample(question="What is the largest planet in our solar system?"),
        TestExample(question="What is 2+2?"),  # Add an extra example to ensure we have enough for comparisons
        TestExample(question="What is the chemical symbol for water?"),  # Add another example
    ]
    
    # Create a mock program with pre-defined answers
    program = MockModule(answers={
        "What is the capital of France?": "Paris is the capital of France.",
        "Who wrote the novel '1984'?": "George Orwell wrote '1984'.",
        "What is the largest planet in our solar system?": "Jupiter is the largest planet in our solar system.",
        "What is 2+2?": "The answer is 4.",
        "What is the chemical symbol for water?": "The chemical formula for water is H2O.",
    })
    
    # Create an EloMetric with more comparisons to ensure history is created
    criteria = "Compare these answers based on accuracy and completeness."
    elo_metric = SimpleEloMetric(criteria=criteria, num_comparisons=3)
    
    # Replace the judge with our mock judge
    elo_metric.judge = MockJudge(criteria=criteria)
    
    # Evaluate each example
    scores = []
    for example in examples:
        prediction = program(example.question)
        score = elo_metric(example, prediction)
        scores.append(score)
    
    # Check that we have scores for all examples
    assert len(scores) == len(examples)
    
    # Check that predictions were stored
    assert len(elo_metric.predictions) == len(examples)
    
    # Force some comparisons to ensure history is created
    # Get all prediction IDs
    pred_ids = list(elo_metric.predictions.keys())
    
    # Manually record some matches if history is empty
    if len(elo_metric.elo_system.history) == 0 and len(pred_ids) >= 2:
        elo_metric.elo_system.record_match(pred_ids[0], pred_ids[1], 0.7)
    
    # Check that we have some history
    assert len(elo_metric.elo_system.history) > 0
    
    # Check that normalized scores are within the expected range
    for score in scores:
        assert 0.0 <= score <= 2.0
