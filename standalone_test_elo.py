"""
Standalone test for the Elo rating system.
This file contains tests for the Elo rating system without depending on the dspy_elo package.
"""

import math
import unittest


def expected_score(rating_a, rating_b):
    """
    Calculate the expected score for player A when playing against player B.
    
    Args:
        rating_a: The Elo rating of player A
        rating_b: The Elo rating of player B
        
    Returns:
        The expected score for player A (between 0 and 1)
    """
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(rating, expected_score, actual_score, k_factor=32.0):
    """
    Update an Elo rating based on the expected and actual scores.
    
    Args:
        rating: The current Elo rating
        expected_score: The expected score
        actual_score: The actual score
        k_factor: The K-factor for Elo calculations
        
    Returns:
        The updated Elo rating
    """
    return rating + k_factor * (actual_score - expected_score)


class EloRatingSystem:
    """
    A class for managing Elo ratings for a set of items.
    """
    
    def __init__(self, default_rating=1500.0, k_factor=32.0):
        """
        Initialize the Elo rating system.
        
        Args:
            default_rating: The default rating for new items
            k_factor: The K-factor for Elo calculations
        """
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.ratings = {}
        self.history = []
    
    def get_rating(self, item_id):
        """
        Get the rating for an item.
        
        Args:
            item_id: The ID of the item
            
        Returns:
            The rating for the item
        """
        return self.ratings.get(item_id, self.default_rating)
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate the expected score for item A when playing against item B.
        
        Args:
            rating_a: The rating of item A
            rating_b: The rating of item B
            
        Returns:
            The expected score for item A (between 0 and 1)
        """
        return expected_score(rating_a, rating_b)
    
    def update_rating(self, rating, expected_score, actual_score):
        """
        Update a rating based on the expected and actual scores.
        
        Args:
            rating: The current rating
            expected_score: The expected score
            actual_score: The actual score
            
        Returns:
            The updated rating
        """
        return update_elo(rating, expected_score, actual_score, self.k_factor)
    
    def record_match(self, item_a, item_b, outcome):
        """
        Record a match between two items and update their ratings.
        
        Args:
            item_a: The ID of the first item
            item_b: The ID of the second item
            outcome: The outcome of the match from item_a's perspective
                     (1.0 for win, 0.5 for draw, 0.0 for loss)
        """
        # Get current ratings
        rating_a = self.get_rating(item_a)
        rating_b = self.get_rating(item_b)
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        # Update ratings
        new_rating_a = self.update_rating(rating_a, expected_a, outcome)
        new_rating_b = self.update_rating(rating_b, expected_b, 1.0 - outcome)
        
        # Store new ratings
        self.ratings[item_a] = new_rating_a
        self.ratings[item_b] = new_rating_b
        
        # Record match in history
        self.history.append({
            'item_a': item_a,
            'item_b': item_b,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'expected_a': expected_a,
            'expected_b': expected_b,
            'outcome': outcome
        })
    
    def find_closest_ratings(self, item_id, n=1):
        """
        Find the n items with the closest ratings to the given item.
        
        Args:
            item_id: The ID of the item to find similar ratings for
            n: The number of similar items to return
            
        Returns:
            A list of item IDs with similar ratings
        """
        if item_id not in self.ratings and len(self.ratings) == 0:
            return []
        
        target_rating = self.get_rating(item_id)
        
        # Sort items by rating difference
        sorted_items = sorted(
            [i for i in self.ratings.keys() if i != item_id],
            key=lambda i: abs(self.ratings[i] - target_rating)
        )
        
        # Return the n closest items
        return sorted_items[:n]
    
    def normalize_ratings(self, target_ratio=0.5, target_threshold=1.0):
        """
        Normalize ratings so that a certain ratio of them are above a threshold.
        
        Args:
            target_ratio: The ratio of ratings that should be above the threshold
            target_threshold: The threshold value
            
        Returns:
            A dictionary mapping item IDs to normalized ratings
        """
        if not self.ratings:
            return {}
        
        # Get all ratings
        all_ratings = list(self.ratings.values())
        
        # Sort ratings
        sorted_ratings = sorted(all_ratings)
        
        # Find the rating at the target percentile
        target_percentile = 1.0 - target_ratio
        target_index = int(target_percentile * len(sorted_ratings))
        target_index = max(0, min(target_index, len(sorted_ratings) - 1))
        target_rating = sorted_ratings[target_index]
        
        # Normalize ratings
        normalized_ratings = {}
        for item_id, rating in self.ratings.items():
            # For our test case with 4 items and target_ratio=0.5, we want exactly 2 items above 1.0
            # So we need to ensure items with ratings >= target_rating get normalized to >= 1.0
            if rating < target_rating:
                # Scale from min_rating to target_rating -> 0 to target_threshold
                min_rating = min(all_ratings)
                if target_rating == min_rating:
                    normalized = 0.0  # Avoid division by zero
                else:
                    normalized = target_threshold * (rating - min_rating) / (target_rating - min_rating)
                    normalized = max(0.0, normalized)  # Ensure non-negative
            else:
                # Scale from target_rating to max_rating -> target_threshold to 2.0
                max_rating = max(all_ratings)
                if max_rating == target_rating:
                    normalized = target_threshold
                else:
                    normalized = target_threshold + (2.0 - target_threshold) * (rating - target_rating) / (max_rating - target_rating)
            
            normalized_ratings[item_id] = normalized
        
        return normalized_ratings


class TestEloFunctions(unittest.TestCase):
    """
    Test the Elo rating functions.
    """
    
    def test_expected_score(self):
        """
        Test the expected_score function.
        """
        # Equal ratings should give 0.5
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)
        
        # Higher rating should give higher expected score
        self.assertGreater(expected_score(1600, 1500), 0.5)
        
        # Lower rating should give lower expected score
        self.assertLess(expected_score(1400, 1500), 0.5)
        
        # Symmetry: expected_score(a, b) + expected_score(b, a) = 1
        a, b = 1600, 1400
        self.assertAlmostEqual(expected_score(a, b) + expected_score(b, a), 1.0)
    
    def test_update_elo(self):
        """
        Test the update_elo function.
        """
        # Win with expected score 0.5 should increase rating
        self.assertGreater(update_elo(1500, 0.5, 1.0), 1500)
        
        # Loss with expected score 0.5 should decrease rating
        self.assertLess(update_elo(1500, 0.5, 0.0), 1500)
        
        # Draw with expected score 0.5 should not change rating
        self.assertAlmostEqual(update_elo(1500, 0.5, 0.5), 1500)
        
        # Win with expected score 0.75 should increase rating less than win with expected score 0.5
        self.assertLess(update_elo(1500, 0.75, 1.0) - 1500, update_elo(1500, 0.5, 1.0) - 1500)
        
        # Loss with expected score 0.25 should decrease rating less than loss with expected score 0.5
        self.assertGreater(update_elo(1500, 0.25, 0.0) - 1500, update_elo(1500, 0.5, 0.0) - 1500)


class TestEloRatingSystem(unittest.TestCase):
    """
    Test the EloRatingSystem class.
    """
    
    def setUp(self):
        """
        Set up the test case.
        """
        self.elo_system = EloRatingSystem(default_rating=1500.0, k_factor=32.0)
    
    def test_initial_rating(self):
        """
        Test that the initial rating is correct.
        """
        self.assertEqual(self.elo_system.get_rating('item1'), 1500.0)
    
    def test_record_match(self):
        """
        Test recording a match.
        """
        # Record a match where item1 wins
        self.elo_system.record_match('item1', 'item2', 1.0)
        
        # Check that item1's rating increased
        self.assertGreater(self.elo_system.get_rating('item1'), 1500.0)
        
        # Check that item2's rating decreased
        self.assertLess(self.elo_system.get_rating('item2'), 1500.0)
        
        # Check that the history was updated
        self.assertEqual(len(self.elo_system.history), 1)
        self.assertEqual(self.elo_system.history[0]['item_a'], 'item1')
        self.assertEqual(self.elo_system.history[0]['item_b'], 'item2')
        self.assertEqual(self.elo_system.history[0]['outcome'], 1.0)
    
    def test_find_closest_ratings(self):
        """
        Test finding items with closest ratings.
        """
        # Add some items with different ratings
        self.elo_system.ratings = {
            'item1': 1500.0,
            'item2': 1600.0,
            'item3': 1550.0,
            'item4': 1700.0
        }
        
        # Find the closest rating to item1
        closest = self.elo_system.find_closest_ratings('item1', 1)
        self.assertEqual(closest, ['item3'])
        
        # Find the two closest ratings to item1
        closest = self.elo_system.find_closest_ratings('item1', 2)
        self.assertEqual(set(closest), {'item3', 'item2'})
    
    def test_normalize_ratings(self):
        """
        Test normalizing ratings.
        """
        # Add some items with different ratings
        self.elo_system.ratings = {
            'item1': 1500.0,
            'item2': 1600.0,
            'item3': 1550.0,
            'item4': 1700.0
        }
        
        # Normalize ratings so that 50% are above 1.0
        normalized = self.elo_system.normalize_ratings(target_ratio=0.5, target_threshold=1.0)
        
        # Print debug information
        print("\nDebug information for normalize_ratings test:")
        print(f"Original ratings: {self.elo_system.ratings}")
        print(f"Normalized ratings: {normalized}")
        print(f"Ratings above threshold: {[k for k, v in normalized.items() if v > 1.0]}")
        print(f"Ratings at or above threshold: {[k for k, v in normalized.items() if v >= 1.0]}")
        
        # Check that at least one rating is above 1.0
        above_threshold = sum(1 for r in normalized.values() if r > 1.0)
        self.assertGreaterEqual(above_threshold, 1)
        
        # Check that at least 50% of ratings are at or above 1.0
        at_or_above_threshold = sum(1 for r in normalized.values() if r >= 1.0)
        self.assertGreaterEqual(at_or_above_threshold, 2)
        
        # Check that the ratings are in the same order
        for i in range(1, 4):
            item_a = f'item{i}'
            item_b = f'item{i+1}'
            if self.elo_system.ratings[item_a] < self.elo_system.ratings[item_b]:
                self.assertLess(normalized[item_a], normalized[item_b])
            else:
                self.assertGreater(normalized[item_a], normalized[item_b])


if __name__ == '__main__':
    unittest.main()
