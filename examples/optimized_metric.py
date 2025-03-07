"""
Example showing how to optimize the EloMetric for better performance.
"""

import dspy
from dspy_elo import EloMetric
from dataclasses import dataclass
from typing import List


# Define a simple example class
@dataclass
class SummaryExample:
    document: str
    summary: str = None
    question: str = "Summarize the following document:"  # For compatibility with EloMetric


# Define a simple DSPy program for summarization
class Summarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("document -> summary")
    
    def forward(self, document):
        return self.summarize(document=document)


def create_dataset() -> List[SummaryExample]:
    """Create a small dataset for summarization."""
    documents = [
        """
        The Elo rating system is a method for calculating the relative skill levels of players in zero-sum games such as chess. 
        It is named after its creator Arpad Elo, a Hungarian-American physics professor. The Elo system was originally invented 
        as an improved chess-rating system over the previously used Harkness system, but is also used as a rating system in 
        many other games and sports.
        """,
        
        """
        Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with 
        the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming 
        paradigms, including structured, object-oriented and functional programming. It is often described as a "batteries included" 
        language due to its comprehensive standard library.
        """,
        
        """
        Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation 
        learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, 
        deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and transformers 
        have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, 
        bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they 
        have produced results comparable to and in some cases surpassing human expert performance.
        """
    ]
    
    return [SummaryExample(document=doc) for doc in documents]


def create_optimization_examples():
    """Create examples for optimizing the judge."""
    return [
        {
            "question": "Summarize the following document:",
            "answer_a": "The Elo rating system calculates relative skill in zero-sum games like chess. Named after Arpad Elo, it improved upon the Harkness system and is used in many games and sports.",
            "answer_b": "Elo is a rating system for chess players created by Arpad Elo.",
            "criteria": "Compare these summaries based on completeness, accuracy, and conciseness.",
            "winner": "A",
            "explanation": "Answer A is more complete, covering the purpose, creator, and applications of the Elo system, while still being concise. Answer B is too brief and misses key information."
        },
        {
            "question": "Summarize the following document:",
            "answer_a": "Python is a programming language that uses indentation and has a standard library.",
            "answer_b": "Python is a high-level, general-purpose programming language emphasizing code readability through significant indentation. It's dynamically typed, garbage-collected, and supports multiple programming paradigms including structured, object-oriented and functional programming. It's known for its comprehensive standard library.",
            "criteria": "Compare these summaries based on completeness, accuracy, and conciseness.",
            "winner": "B",
            "explanation": "Answer B provides a more comprehensive overview of Python, covering its key characteristics, typing system, and programming paradigms. Answer A is too simplistic and misses important details."
        }
    ]


def main():
    # Set up DSPy
    lm = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=lm)
    
    # Create a dataset
    dataset = create_dataset()
    
    # Create a DSPy program
    summarizer = Summarizer()
    
    # Define criteria for comparing summaries
    criteria = """
    Compare these two summaries based on:
    1. Completeness - Which summary captures more key information from the original document?
    2. Accuracy - Which summary is more factually accurate compared to the original document?
    3. Conciseness - Which summary expresses the information more efficiently without unnecessary details?
    4. Clarity - Which summary is more clearly written and easier to understand?
    
    Choose the better summary based on these criteria.
    """
    
    # Create an Elo metric
    elo_metric = EloMetric(
        criteria=criteria,
        num_comparisons=2,
        normalize_ratio=0.5,
        question_field="question",  # Field in example containing the question
        answer_field="summary"      # Field in prediction containing the answer
    )
    
    # Optimize the metric using a few examples
    print("Optimizing the EloMetric judge...")
    optimization_examples = create_optimization_examples()
    optimizer = dspy.teleprompt.BootstrapFewShot(metric=lambda x, y: 1.0)  # Simple optimizer
    optimized_metric = elo_metric.optimize(optimization_examples, optimizer)
    
    # Evaluate the program with the optimized metric
    print("Evaluating the summarizer with the optimized metric...")
    evaluator = dspy.evaluate.Evaluate(
        devset=dataset,
        metric=optimized_metric,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    results = evaluator(summarizer)
    
    print(f"\nAverage score: {results['metric']:.4f}")
    
    # Print the Elo ratings
    print("\nElo ratings and normalized scores:")
    for pred_id, rating in optimized_metric.elo_system.ratings.items():
        example = optimized_metric.examples[pred_id]
        prediction = optimized_metric.predictions[pred_id]
        normalized_score = optimized_metric.normalize_score(rating)
        
        print(f"Document: {example.document[:50]}...")
        print(f"Summary: {prediction.summary}")
        print(f"Elo Rating: {rating:.2f}")
        print(f"Normalized Score: {normalized_score:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
