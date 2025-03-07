"""
Integration test for the EloMetric.
"""

import pytest
from unittest.mock import MagicMock, patch
import dspy
from dataclasses import dataclass
from dspy_elo import EloMetric


# Create a simple example and prediction class for testing
@dataclass
class TestExample:
    question: str
    answer: str = None


class MockDSPyModule(dspy.Module):
    """Mock DSPy module for testing."""
    
    def __init__(self, answers=None):
        super().__init__()
        self.answers = answers or {}
    
    def forward(self, question):
        # Return a mock prediction with a pre-defined answer
        mock_pred = MagicMock()
        mock_pred.answer = self.answers.get(question, "Default answer")
        return mock_pred


@pytest.fixture
def mock_dspy_predict():
    """Mock the dspy.Predict class."""
    with patch('dspy.Predict') as mock_predict:
        # Create a mock instance
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
        
        # Make the mock_predict return the mock_instance when called
        mock_predict.return_value = mock_instance
        yield mock_instance


def test_integration_with_mock_dspy(mock_dspy_predict):
    """Test integration with a mock DSPy program."""
    # Create examples
    examples = [
        TestExample(question="What is the capital of France?"),
        TestExample(question="Who wrote the novel '1984'?"),
        TestExample(question="What is the largest planet in our solar system?"),
    ]
    
    # Create a mock DSPy program with pre-defined answers
    program = MockDSPyModule(answers={
        "What is the capital of France?": "Paris is the capital of France.",
        "Who wrote the novel '1984'?": "George Orwell wrote '1984'.",
        "What is the largest planet in our solar system?": "Jupiter is the largest planet in our solar system.",
    })
    
    # Create an EloMetric
    criteria = "Compare these answers based on accuracy and completeness."
    elo_metric = EloMetric(criteria=criteria, num_comparisons=2)
    
    # Replace the judge's predictor with our mock
    elo_metric.judge.predictor = mock_dspy_predict
    
    # Evaluate each example
    scores = []
    for example in examples:
        prediction = program(example.question)
        score = elo_metric(example, prediction)
        scores.append(score)
    
    # Check that we have scores for all examples
    assert len(scores) == len(examples)
    
    # Check that ratings were updated
    assert len(elo_metric.elo_system.ratings) == len(examples)
    
    # Check that we have some history
    assert len(elo_metric.elo_system.history) > 0
    
    # Check that normalized scores are within the expected range
    for score in scores:
        assert 0.0 <= score <= 2.0
