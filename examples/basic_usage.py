"""
Basic usage example for the DSPy Elo metric.
"""

import dspy
from dspy_elo import EloMetric
from dataclasses import dataclass


# Define a simple example class
@dataclass
class QAExample:
    question: str
    answer: str = None


# Define a simple DSPy program
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.generate_answer(question=question)


def main():
    # Set up DSPy
    lm = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=lm)
    
    # Create a dataset
    questions = [
        "What is the capital of France?",
        "Who wrote the novel '1984'?",
        "What is the largest planet in our solar system?",
        "What is the chemical symbol for gold?",
        "Who painted the Mona Lisa?",
    ]
    
    dataset = [QAExample(question=q) for q in questions]
    
    # Create a DSPy program
    qa_program = SimpleQA()
    
    # Define criteria for comparing answers
    criteria = """
    Compare these two answers based on:
    1. Accuracy - Which answer is more factually correct?
    2. Completeness - Which answer addresses more aspects of the question?
    3. Clarity - Which answer is more clearly written and easier to understand?
    
    Choose the better answer based on these criteria.
    """
    
    # Create an Elo metric
    elo_metric = EloMetric(
        criteria=criteria,
        num_comparisons=3,
        normalize_ratio=0.5
    )
    
    # Evaluate the program
    evaluator = dspy.evaluate.Evaluate(
        devset=dataset,
        metric=elo_metric,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    results = evaluator(qa_program)
    
    print(f"Average score: {results['metric']:.4f}")
    
    # Print the Elo ratings
    print("\nElo ratings:")
    for pred_id, rating in elo_metric.elo_system.ratings.items():
        example = elo_metric.examples[pred_id]
        prediction = elo_metric.predictions[pred_id]
        print(f"Q: {example.question}")
        print(f"A: {prediction.answer}")
        print(f"Rating: {rating:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
