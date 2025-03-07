"""
Direct test for the EloMetric class.
"""

import unittest
from dataclasses import dataclass
import uuid
import sys
import os

# Add the parent directory to the path so we can import from fully_isolated_test_elo_metric_direct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the isolated implementation
from fully_isolated_test_elo_metric_direct import SimpleEloMetric


@dataclass
class SimpleExample:
    """Simple example for testing."""
    question: str
    answer: str = None


@dataclass
class SimplePrediction:
    """Simple prediction for testing."""
    answer: str


class TestEloMetricDirect(unittest.TestCase):
    """Test the EloMetric class directly."""
    
    def setUp(self):
        """Set up the test case."""
        self.metric = SimpleEloMetric(
            criteria="Test criteria",
            num_comparisons=2,
            normalize_ratio=0.5
        )
    
    def test_basic_functionality(self):
        """Test basic functionality of the EloMetric."""
        # Create examples and predictions
        example1 = SimpleExample(question="What is the capital of France?")
        example2 = SimpleExample(question="What is the capital of France?")
        example3 = SimpleExample(question="What is the capital of France?")
        
        prediction1 = SimplePrediction(answer="Paris")
        prediction2 = SimplePrediction(answer="Paris is the capital of France.")
        prediction3 = SimplePrediction(answer="The capital of France is Paris.")
        
        # Test the metric
        score1 = self.metric(example1, prediction1)
        score2 = self.metric(example2, prediction2)
        score3 = self.metric(example3, prediction3)
        
        # Print debug information
        print(f"Scores: {score1}, {score2}, {score3}")
        print(f"Ratings: {list(self.metric.elo_system.ratings.values())}")
        
        # Basic assertions
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 2.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 2.0)
        self.assertGreaterEqual(score3, 0.0)
        self.assertLessEqual(score3, 2.0)


if __name__ == "__main__":
    unittest.main()
