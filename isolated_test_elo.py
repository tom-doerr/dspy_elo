"""
Isolated test for the Elo rating system without DSPy dependencies.
"""

import sys
import unittest
import numpy as np

# Import the Elo class directly from the file
sys.path.append('/home/tom/git/dspy_elo')
from dspy_elo.elo import EloRatingSystem, expected_score, update_elo


class TestElo(unittest.TestCase):
    """Test the Elo rating system."""
    
    def test_expected_score(self):
        """Test the expected_score function."""
        # Equal ratings should give 0.5
        self.assertEqual(expected_score(1500, 1500), 0.5)
        
        # Higher rating should give higher expected score
        self.assertGreater(expected_score(1600, 1500), 0.5)
        self.assertLess(expected_score(1500, 1600), 0.5)
        
        # Verify specific value from Elo formula
        self.assertLess(abs(expected_score(1613, 1609) - 0.506), 0.001)

    def test_update_elo(self):
        """Test the update_elo function."""
        # No change if actual equals expected
        self.assertEqual(update_elo(1500, 0.5, 0.5, k_factor=32), 1500)
        
        # Increase rating for better than expected performance
        self.assertGreater(update_elo(1500, 0.5, 1.0, k_factor=32), 1500)
        
        # Decrease rating for worse than expected performance
        self.assertLess(update_elo(1500, 0.5, 0.0, k_factor=32), 1500)
        
        # Verify specific value from Elo formula
        # If rating is 1613, expected score is 2.867, actual score is 2.5, k=32
        # New rating should be 1601
        self.assertLess(abs(update_elo(1613, 2.867/5, 2.5/5, k_factor=32) - 1601), 1)

    def test_elo_rating_system_init(self):
        """Test EloRatingSystem initialization."""
        elo_system = EloRatingSystem(default_rating=1500, k_factor=32)
        self.assertEqual(elo_system.default_rating, 1500)
        self.assertEqual(elo_system.k_factor, 32)
        self.assertEqual(len(elo_system.ratings), 0)
        self.assertEqual(len(elo_system.history), 0)

    def test_get_rating(self):
        """Test get_rating method."""
        elo_system = EloRatingSystem(default_rating=1500)
        
        # Should return default rating for new items
        self.assertEqual(elo_system.get_rating("player1"), 1500)
        
        # Should return stored rating for existing items
        elo_system.ratings["player1"] = 1600
        self.assertEqual(elo_system.get_rating("player1"), 1600)

    def test_record_match(self):
        """Test record_match method."""
        elo_system = EloRatingSystem(default_rating=1500, k_factor=32)
        
        # Record a match where player1 wins
        new_rating_a, new_rating_b = elo_system.record_match("player1", "player2", 1.0)
        
        # Winner's rating should increase
        self.assertGreater(new_rating_a, 1500)
        # Loser's rating should decrease
        self.assertLess(new_rating_b, 1500)
        
        # Ratings should be stored
        self.assertEqual(elo_system.ratings["player1"], new_rating_a)
        self.assertEqual(elo_system.ratings["player2"], new_rating_b)
        
        # History should be updated
        self.assertEqual(len(elo_system.history), 1)
        self.assertEqual(elo_system.history[0]["item_a"], "player1")
        self.assertEqual(elo_system.history[0]["item_b"], "player2")
        self.assertEqual(elo_system.history[0]["outcome"], 1.0)

    def test_find_closest_ratings(self):
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
        self.assertEqual(len(closest), 2)
        self.assertIn("player5", closest)  # 1450
        self.assertIn("player3", closest)  # 1400
        
        # Find closest to player4 (1700)
        closest = elo_system.find_closest_ratings("player4", n=2)
        self.assertEqual(len(closest), 2)
        self.assertIn("player2", closest)  # 1600
        
        # Find closest to new player (should use default rating 1500)
        closest = elo_system.find_closest_ratings("new_player", n=2)
        self.assertEqual(len(closest), 2)
        self.assertIn("player1", closest)  # 1500
        
        # Empty ratings
        elo_system.ratings = {}
        self.assertEqual(elo_system.find_closest_ratings("player1"), [])

    def test_normalize_ratings(self):
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
        
        # Debug information
        print("\nDebug information for normalize_ratings test:")
        print(f"Original ratings: {elo_system.ratings}")
        print(f"Normalized ratings: {normalized}")
        
        # Count how many are above 1.0
        above_threshold = [pid for pid, rating in normalized.items() if rating > 1.0]
        at_or_above_threshold = [pid for pid, rating in normalized.items() if rating >= 1.0]
        
        print(f"Ratings above threshold: {above_threshold}")
        print(f"Ratings at or above threshold: {at_or_above_threshold}")
        
        # 40% of 5 = 2
        self.assertEqual(len(above_threshold), 2)
        
        # The threshold should be at the 60th percentile
        # Sorted ratings: 1300, 1400, 1500, 1600, 1700
        # 60th percentile is at index 2 (value 1500)
        # So 1500 should be normalized to 1.0
        self.assertLess(abs(normalized["player1"] - 1.0), 0.001)
        
        # Higher ratings should be above 1.0
        self.assertGreater(normalized["player2"], 1.0)
        self.assertGreater(normalized["player4"], 1.0)
        
        # Lower ratings should be below 1.0
        self.assertLess(normalized["player3"], 1.0)
        self.assertLess(normalized["player5"], 1.0)
        
        # Empty ratings
        elo_system.ratings = {}
        self.assertEqual(elo_system.normalize_ratings(), {})


if __name__ == "__main__":
    unittest.main()
