"""
DSPy Elo - An Elo-based metric for DSPy.
"""

try:
    from .elo_metric_new import EloMetric
except ImportError:
    # Fallback to core implementation if DSPy is not available
    from .core.metric_base import BaseEloMetric as EloMetric

# Also expose core components for advanced usage
from .core import EloRatingSystem, expected_score, update_elo, BaseEloMetric
from .core.simple_judge import SimpleJudge

__all__ = [
    "EloMetric",
    "EloRatingSystem",
    "expected_score",
    "update_elo",
    "BaseEloMetric",
    "SimpleJudge",
]
