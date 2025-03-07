"""
Mock classes for testing without DSPy dependencies.
"""
from dataclasses import dataclass

class MockExample:
    """A mock example class for testing."""
    def __init__(self, question, reference=""):
        self.question = question
        self.reference = reference


class MockPrediction:
    """A mock prediction class for testing."""
    def __init__(self, answer):
        self.answer = answer


class MockExampleDict(dict):
    """A mock example class that inherits from dict for testing."""
    pass


class MockPredictionDict(dict):
    """A mock prediction class that inherits from dict for testing."""
    pass


# Legacy test classes with dataclass - these will cause pytest warnings
# but are kept for backward compatibility with existing tests
@dataclass
class TestExample:
    """A test example class using dataclass for backward compatibility."""
    question: str


@dataclass
class TestPrediction:
    """A test prediction class using dataclass for backward compatibility."""
    answer: str
