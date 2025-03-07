"""
Mock classes for testing without using dataclasses to avoid pytest collection warnings.
"""

from typing import Dict, Any, List, Optional

class MockExample:
    """Mock example class for testing."""
    def __init__(self, question: str, expected: str):
        self.question = question
        self.expected = expected

class MockPrediction:
    """Mock prediction class for testing."""
    def __init__(self, answer: str):
        self.answer = answer

class MockExampleDict(dict):
    """Mock example class that behaves like a dictionary."""
    def __init__(self, question: str, expected: str):
        super().__init__(question=question, expected=expected)

class MockPredictionDict(dict):
    """Mock prediction class that behaves like a dictionary."""
    def __init__(self, answer: str):
        super().__init__(answer=answer)

class MockEloMetric:
    """Mock Elo metric class for testing."""
    def __init__(
        self,
        num_comparisons: int = 5,
        normalize_ratio: float = 0.5,
        default_rating: float = 1500.0,
        k_factor: float = 32.0,
        min_score: float = 0.0,
        max_score: float = 2.0,
        question_field: str = "question",
        answer_field: str = "answer",
    ):
        # Validate parameters
        if num_comparisons < 0:
            raise ValueError("num_comparisons must be non-negative")
        
        if normalize_ratio < 0 or normalize_ratio > 1:
            raise ValueError("normalize_ratio must be between 0 and 1")
        
        if min_score > max_score:
            raise ValueError("min_score must be less than or equal to max_score")
        
        self.num_comparisons = num_comparisons
        self.normalize_ratio = normalize_ratio
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.min_score = min_score
        self.max_score = max_score
        self.question_field = question_field
        self.answer_field = answer_field
        
        # Store examples and predictions
        self.examples = {}
        self.predictions = {}
        
        # Track number of comparisons performed
        self.comparison_count = 0
    
    def __call__(self, example: Any, prediction: Any, trace: Optional[Any] = None) -> float:
        """
        Evaluate a prediction against an example.
        
        Args:
            example: Example to evaluate against
            prediction: Prediction to evaluate
            trace: Optional trace information
            
        Returns:
            Score for the prediction, or a boolean if trace is provided
        """
        # Validate inputs
        if example is None:
            raise ValueError("Example cannot be None")
        
        if prediction is None:
            raise ValueError("Prediction cannot be None")
        
        # Generate a unique ID for this prediction
        pred_id = f"pred_{len(self.predictions)}"
        
        # Store example and prediction
        self.examples[pred_id] = example
        self.predictions[pred_id] = prediction
        
        # Get question and answer
        question = self._extract_field(example, self.question_field, str(example))
        answer = self._extract_field(prediction, self.answer_field, str(prediction))
        
        # Return a mock score
        score = (self.min_score + self.max_score) / 2.0
        
        # For optimization, return a boolean based on whether the score is above 1.0
        if trace is not None:
            return score > 1.0
        
        return score
    
    def _extract_field(self, obj: Any, field: str, default: Any) -> Any:
        """
        Extract a field from an object, handling different object types.
        
        Args:
            obj: Object to extract field from
            field: Field name to extract
            default: Default value if field is not found
            
        Returns:
            Value of the field, or default if not found
        """
        try:
            return getattr(obj, field)
        except (AttributeError, TypeError):
            try:
                return obj[field]
            except (KeyError, TypeError):
                return default
