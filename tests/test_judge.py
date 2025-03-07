"""
Tests for the DSPy-based judge.
"""

import pytest
from unittest.mock import MagicMock, patch
import dspy
from dspy_elo.judge import AnswerJudge, CompareAnswers


@pytest.fixture
def mock_dspy_predict():
    """Mock the dspy.Predict class."""
    with patch('dspy.Predict') as mock_predict:
        # Create a mock instance
        mock_instance = MagicMock()
        # Make the mock_predict return the mock_instance when called
        mock_predict.return_value = mock_instance
        yield mock_instance


def test_answer_judge_init():
    """Test AnswerJudge initialization."""
    criteria = "Test criteria"
    judge = AnswerJudge(criteria=criteria)
    
    assert judge.criteria == criteria
    assert isinstance(judge.predictor, dspy.Predict)


def test_compare_winner_a(mock_dspy_predict):
    """Test compare method when A wins."""
    # Set up the mock to return a result with winner 'A'
    mock_result = MagicMock()
    mock_result.winner = "A"
    mock_result.explanation = "A is better"
    mock_dspy_predict.return_value = mock_result
    
    # Create the judge
    judge = AnswerJudge(criteria="Test criteria")
    judge.predictor = mock_dspy_predict
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer B"
    )
    
    # Check the results
    assert winner == "A"
    assert explanation == "A is better"
    assert outcome == 1.0
    
    # Verify the mock was called with the right arguments
    mock_dspy_predict.assert_called_once_with(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer B",
        criteria="Test criteria"
    )


def test_compare_winner_b(mock_dspy_predict):
    """Test compare method when B wins."""
    # Set up the mock to return a result with winner 'B'
    mock_result = MagicMock()
    mock_result.winner = "B"
    mock_result.explanation = "B is better"
    mock_dspy_predict.return_value = mock_result
    
    # Create the judge
    judge = AnswerJudge(criteria="Test criteria")
    judge.predictor = mock_dspy_predict
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer B"
    )
    
    # Check the results
    assert winner == "B"
    assert explanation == "B is better"
    assert outcome == 0.0


def test_compare_tie(mock_dspy_predict):
    """Test compare method when it's a tie."""
    # Set up the mock to return a result with winner 'TIE'
    mock_result = MagicMock()
    mock_result.winner = "TIE"
    mock_result.explanation = "Both are equal"
    mock_dspy_predict.return_value = mock_result
    
    # Create the judge
    judge = AnswerJudge(criteria="Test criteria")
    judge.predictor = mock_dspy_predict
    
    # Compare answers
    winner, explanation, outcome = judge.compare(
        question="Test question",
        answer_a="Answer A",
        answer_b="Answer B"
    )
    
    # Check the results
    assert winner == "TIE"
    assert explanation == "Both are equal"
    assert outcome == 0.5
