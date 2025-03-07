"""
Implementation of the Elo rating system for DSPy.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score of player A in a match against player B.
    
    Args:
        rating_a: Elo rating for player A
        rating_b: Elo rating for player B
        
    Returns:
        Expected score for player A (between 0 and 1)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(old_rating: float, expected: float, actual: float, k_factor: float = 32.0) -> float:
    """
    Calculate the new Elo rating for a player.
    
    Args:
        old_rating: The previous Elo rating
        expected: The expected score for this match
        actual: The actual score for this match (1 for win, 0.5 for draw, 0 for loss)
        k_factor: The k-factor for Elo (default: 32)
        
    Returns:
        The updated Elo rating
    """
    return old_rating + k_factor * (actual - expected)


class EloRatingSystem:
    """
    A class to manage Elo ratings for a collection of items.
    """
    
    def __init__(self, default_rating: float = 1500.0, k_factor: float = 32.0):
        """
        Initialize the Elo rating system.
        
        Args:
            default_rating: The default rating for new items
            k_factor: The k-factor for Elo calculations
        """
        self.ratings: Dict[str, float] = {}
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.history: List[Dict] = []
    
    def get_rating(self, item_id: str) -> float:
        """
        Get the current rating for an item.
        
        Args:
            item_id: The identifier for the item
            
        Returns:
            The current Elo rating for the item
        """
        return self.ratings.get(item_id, self.default_rating)
    
    def record_match(self, item_a: str, item_b: str, outcome: float) -> Tuple[float, float]:
        """
        Record the outcome of a match between two items and update their ratings.
        
        Args:
            item_a: The identifier for the first item
            item_b: The identifier for the second item
            outcome: The outcome from item_a's perspective (1 for win, 0.5 for draw, 0 for loss)
            
        Returns:
            A tuple of the new ratings (rating_a, rating_b)
        """
        # Get current ratings
        rating_a = self.get_rating(item_a)
        rating_b = self.get_rating(item_b)
        
        # Calculate expected scores
        expected_a = expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        # Update ratings
        new_rating_a = update_elo(rating_a, expected_a, outcome, self.k_factor)
        new_rating_b = update_elo(rating_b, expected_b, 1.0 - outcome, self.k_factor)
        
        # Store new ratings
        self.ratings[item_a] = new_rating_a
        self.ratings[item_b] = new_rating_b
        
        # Record history
        self.history.append({
            'item_a': item_a,
            'item_b': item_b,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'outcome': outcome
        })
        
        return new_rating_a, new_rating_b
    
    def find_closest_ratings(self, item_id: str, n: int = 5) -> List[str]:
        """
        Find the n items with the closest ratings to the given item.
        
        Args:
            item_id: The identifier for the reference item
            n: The number of items to return
            
        Returns:
            A list of item identifiers with the closest ratings
        """
        if item_id not in self.ratings and len(self.ratings) == 0:
            return []
        
        target_rating = self.get_rating(item_id)
        
        # Get all items except the target
        other_items = [(id, rating) for id, rating in self.ratings.items() if id != item_id]
        
        # Sort by rating difference, then by rating
        other_items.sort(key=lambda x: (abs(x[1] - target_rating), x[1]))
        
        # Return the n closest items
        return [id for id, _ in other_items[:n]]
    
    def normalize_ratings(self, target_ratio: float = 0.5, target_threshold: float = 1.0) -> Dict[str, float]:
        """
        Normalize ratings so that a specified ratio of items have ratings above a threshold.
        
        Args:
            target_ratio: The desired ratio of items with ratings above the threshold
            target_threshold: The threshold value (default: 1.0)
            
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
