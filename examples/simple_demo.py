"""
Simple demonstration of the DSPy Elo metric.
"""

import dspy
from dspy_elo import EloMetric
from dataclasses import dataclass


# Define a simple example class
@dataclass
class QAExample:
    question: str
    answer: str = None


def main():
    print("DSPy Elo Metric Demonstration")
    print("-----------------------------")
    
    # Create an Elo metric
    criteria = """
    Compare these two answers based on:
    1. Accuracy - Which answer is more factually correct?
    2. Completeness - Which answer addresses more aspects of the question?
    3. Clarity - Which answer is more clearly written and easier to understand?
    
    Choose the better answer based on these criteria.
    """
    
    elo_metric = EloMetric(
        criteria=criteria,
        num_comparisons=2,
        normalize_ratio=0.5
    )
    
    # Create some examples and predictions
    examples = [
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
        QAExample(question="What is the capital of France?"),
    ]
    
    # Create mock predictions
    class MockPrediction:
        def __init__(self, answer):
            self.answer = answer
    
    predictions = [
        MockPrediction("Paris is the capital of France."),
        MockPrediction("The capital of France is Paris, a city known for its art, culture, and the Eiffel Tower."),
        MockPrediction("Paris.")
    ]
    
    # Mock the judge to avoid needing a real LLM
    def mock_compare(question, answer_a, answer_b):
        # Simple logic: longer answer wins
        if len(answer_a) > len(answer_b):
            return "A", "A is more detailed", 1.0
        elif len(answer_b) > len(answer_a):
            return "B", "B is more detailed", 0.0
        else:
            return "TIE", "Both are equally detailed", 0.5
    
    # Replace the judge's compare method
    elo_metric.judge.compare = mock_compare
    
    # Evaluate each example
    print("\nEvaluating predictions...")
    scores = []
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"\nPrediction {i+1}:")
        print(f"Question: {example.question}")
        print(f"Answer: {prediction.answer}")
        
        score = elo_metric(example, prediction)
        scores.append(score)
        print(f"Score: {score:.4f}")
    
    print("\nFinal Elo ratings:")
    for pred_id, rating in elo_metric.elo_system.ratings.items():
        normalized_score = elo_metric.normalize_score(rating)
        print(f"ID: {pred_id}")
        print(f"Rating: {rating:.2f}")
        print(f"Normalized Score: {normalized_score:.4f}")
        print("-" * 40)
    
    print(f"\nAverage score: {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
