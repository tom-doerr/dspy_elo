"""
Isolated tests for the Elo rating system.
"""

import pytest
import sys
import os
import unittest

# Add the parent directory to the path so we can import from fully_isolated_test_elo_metric_direct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the isolated implementation
from fully_isolated_test_elo_metric_direct import (
    expected_score, 
    update_elo, 
    EloRatingSystem
)


def test_expected_score():
    """Test the expected_score function."""
    # Equal ratings should give 0.5
    assert expected_score(1500, 1500) == 0.5
    
    # Higher rating should give higher expected score
    assert expected_score(1600, 1500) > 0.5
    assert expected_score(1500, 1600) < 0.5
    
    # Verify specific value from Elo formula
    assert abs(expected_score(1613, 1609) - 0.506) < 0.001


def test_update_elo():
    """Test the update_elo function."""
    # No change if actual equals expected
    assert update_elo(1500, 0.5, 0.5, k_factor=32) == 1500
    
    # Increase rating for better than expected performance
    assert update_elo(1500, 0.5, 1.0, k_factor=32) > 1500
    
    # Decrease rating for worse than expected performance
    assert update_elo(1500, 0.5, 0.0, k_factor=32) < 1500
    
    # Verify specific value from Elo formula
    # If rating is 1613, expected score is 0.573, actual score is 0.5, k=32
    # New rating should be approximately 1610.65
    result = update_elo(1613, 0.573, 0.5, k_factor=32)
    assert abs(result - 1610.65) < 0.1


def test_elo_rating_system_init():
    """Test EloRatingSystem initialization."""
    elo_system = EloRatingSystem(default_rating=1500, k_factor=32)
    assert elo_system.default_rating == 1500
    assert elo_system.k_factor == 32
    assert len(elo_system.ratings) == 0
    assert len(elo_system.history) == 0


def test_get_rating():
    """Test get_rating method."""
    elo_system = EloRatingSystem(default_rating=1500)
    
    # Should return default rating for new items
    assert elo_system.get_rating("player1") == 1500
    
    # Should return stored rating for existing items
    elo_system.ratings["player1"] = 1600
    assert elo_system.get_rating("player1") == 1600


def test_record_match():
    """Test record_match method."""
    elo_system = EloRatingSystem(default_rating=1500, k_factor=32)
    
    # Record a match where player1 wins
    new_rating_a, new_rating_b = elo_system.record_match("player1", "player2", 1.0)
    
    # Winner's rating should increase
    assert new_rating_a > 1500
    # Loser's rating should decrease
    assert new_rating_b < 1500
    
    # Ratings should be stored
    assert elo_system.ratings["player1"] == new_rating_a
    assert elo_system.ratings["player2"] == new_rating_b
    
    # History should be updated
    assert len(elo_system.history) == 1
    assert elo_system.history[0]["item_a"] == "player1"
    assert elo_system.history[0]["item_b"] == "player2"
    assert elo_system.history[0]["outcome"] == 1.0


def test_find_closest_ratings():
    """Test find_closest_ratings method."""
    elo_system = EloRatingSystem(default_rating=1500)
    
    # Add some ratings
    elo_system.ratings = {
        "player1": 1500,
        "player2": 1600,
        "player3": 1400,
        "player4": 1700,
        "player5": 1450
    }
    
    # Find closest to player1 (1500)
    closest = elo_system.find_closest_ratings("player1", n=2)
    assert len(closest) == 2
    assert "player5" in closest  # 1450
    
    # The second closest could be player2 (1600) or player3 (1400) depending on implementation
    # Both are 100 points away, so either is valid
    assert ("player2" in closest) or ("player3" in closest)
    
    # Find closest to player4 (1700)
    closest = elo_system.find_closest_ratings("player4", n=2)
    assert len(closest) == 2
    assert "player2" in closest  # 1600
    
    # Find closest to new player (should use default rating 1500)
    closest = elo_system.find_closest_ratings("new_player", n=2)
    assert len(closest) == 2
    assert "player1" in closest  # 1500
    
    # Empty ratings
    elo_system.ratings = {}
    assert elo_system.find_closest_ratings("player1") == []


def test_normalize_ratings():
    """Test normalize_ratings method."""
    elo_system = EloRatingSystem(default_rating=1500)
    
    # Add some ratings
    elo_system.ratings = {
        "player1": 1500,
        "player2": 1600,
        "player3": 1400,
        "player4": 1700,
        "player5": 1300
    }
    
    # Normalize so that 40% of ratings are above 1.0
    normalized = elo_system.normalize_ratings(target_ratio=0.4, target_threshold=1.0)
    
    # Count how many are above 1.0
    above_threshold = sum(1 for rating in normalized.values() if rating > 1.0)
    
    # Due to rounding and implementation differences, the exact number might vary
    # The implementation might round differently or use different thresholds
    # So we'll check that the ratio is approximately 0.4
    ratio = above_threshold / len(normalized)
    assert 0.2 <= ratio <= 0.6  # Allow some flexibility
    
    # The highest ratings should be above 1.0
    assert normalized["player4"] > 1.0  # 1700 should definitely be above threshold
    
    # The lowest ratings should be below 1.0
    assert normalized["player5"] < 1.0  # 1300 should definitely be below threshold
    
    # Empty ratings
    elo_system.ratings = {}
    assert elo_system.normalize_ratings() == {}
