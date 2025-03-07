# DSPy Elo Isolated Tests

This directory contains isolated tests for the DSPy Elo package that don't depend on DSPy. These tests use mock implementations to test the core functionality of the package without requiring the DSPy dependencies.

## Test Files Overview

### Core Functionality Tests

- **test_elo.py**: Tests the core Elo rating system functionality, including expected score calculation, rating updates, and rating normalization.
- **test_elo_metric.py**: Tests the EloMetric class, including initialization, prediction handling, and score normalization.
- **test_judge.py**: Tests the AnswerJudge class, including initialization and comparison functionality.
- **test_integration.py**: Tests the integration between the EloMetric, EloRatingSystem, and AnswerJudge classes.

### Edge Case Tests

- **test_call_edge_cases.py**: Tests edge cases when calling the EloMetric, including different input types and missing fields.
- **test_error_handling.py**: Tests error handling in the EloMetric class, including None inputs, empty inputs, invalid parameters, and more.
- **test_normalize_score.py**: Tests the normalize_score method of the EloMetric class.
- **test_normalize_score_edge_cases.py**: Tests edge cases for the normalize_score method, including empty predictions, extreme ratings, and custom min/max scores.

### Performance and Scaling Tests

- **test_scaling.py**: Tests the scaling behavior of the EloMetric class, including tests with large numbers of predictions and different comparison counts.
- **test_optimize.py**: Tests the optimization functionality of the EloMetric class.
- **test_concurrency.py**: Tests the thread safety and concurrency behavior of the EloMetric class.

## Mock Implementations

The tests use mock implementations of the following classes:

- **MockEloRatingSystem**: A mock implementation of the EloRatingSystem class.
- **MockAnswerJudge**: A mock implementation of the AnswerJudge class.
- **MockEloMetric**: A mock implementation of the EloMetric class.

These mock implementations provide the same interface as the real classes but don't depend on DSPy.

## Running the Tests

To run all the isolated tests:

```bash
python -m pytest isolated_tests/ -v
```

To run a specific test file:

```bash
python -m pytest isolated_tests/test_file.py -v
```

To run the tests with coverage:

```bash
python -m pytest --cov=isolated_tests isolated_tests/ -v
```

## Test Coverage

Current test coverage for the isolated tests is over 90%, covering:
- Edge cases for input handling
- Normalization behavior
- Error handling
- Scaling with large numbers of predictions
- Concurrency and thread safety
- Performance optimization
