"""
Core Elo rating system functionality that doesn't depend on DSPy.
"""

from .elo_rating import EloRatingSystem, expected_score, update_elo
from .metric_base import BaseEloMetric
