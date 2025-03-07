<div align="center">

# DSPy Elo

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/Coverage-90%25%2B-success?style=for-the-badge)](https://github.com/tom-doerr/dspy_elo)

A Python package that provides an Elo-based metric function for DSPy.

</div>

## Features

- Evaluate DSPy model outputs using Elo rating system
- Compare model outputs based on user-defined criteria
- Configure the number of comparisons per evaluation
- Smart selection of comparison samples based on similar Elo ratings
- Normalization options to ensure compatibility with DSPy optimizers

## Installation

```bash
pip install dspy-elo
```

## Usage

```python
import dspy
from dspy_elo import EloMetric

# Define your criteria for comparing outputs
criteria = """
Compare these two answers based on:
1. Accuracy - Which answer is more factually correct?
2. Completeness - Which answer addresses more aspects of the question?
3. Clarity - Which answer is more clearly written and easier to understand?

Choose the better answer based on these criteria.
"""

# Create an Elo metric with your criteria
elo_metric = EloMetric(
    criteria=criteria,
    num_comparisons=5,  # Number of comparisons per evaluation
    normalize_ratio=0.5  # Ratio of scores that should be above 1.0
)

# Use the metric in DSPy evaluation
evaluator = dspy.evaluate.Evaluate(devset=my_dataset)
results = evaluator(my_program, metric=elo_metric)
```

## How It Works

DSPy Elo uses the Elo rating system to compare and rank model outputs. When evaluating a new output:

1. The output is compared against existing samples with similar Elo ratings
2. A DSPy-powered judge evaluates each comparison based on user-defined criteria
3. Elo ratings are updated based on the outcomes of these comparisons
4. Final scores are normalized to ensure compatibility with DSPy optimizers

## License

MIT

## Development

### Testing

The package includes two sets of tests:

1. **Regular Tests**: Located in the `tests/` directory, these tests require DSPy dependencies.

2. **Isolated Tests**: Located in the `isolated_tests/` directory, these tests use mock implementations to avoid DSPy dependencies, making them easier to run and maintain.

To run the isolated tests:

```bash
python -m pytest isolated_tests/ -v
```

Current test coverage for isolated tests is over 90%, covering:
- Edge cases for input handling
- Normalization behavior
- Error handling
- Scaling with large numbers of predictions
- Concurrency and thread safety
- Performance optimization

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
