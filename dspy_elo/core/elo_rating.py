"""
Core implementation of the Elo rating system.
"""

import math
import threading
from typing import Dict, List, Tuple, Union, Optional


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score for player A against player B.
    
    Args:
        rating_a: Elo rating of player A
        rating_b: Elo rating of player B
        
    Returns:
        Expected score for player A (between 0 and 1)
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float, k_factor: float = 32.0) -> float:
    """
    Update an Elo rating based on the expected and actual outcomes.
    
    Args:
        rating: Current Elo rating
        expected: Expected outcome (between 0 and 1)
        actual: Actual outcome (0 for loss, 0.5 for draw, 1 for win)
        k_factor: K-factor for Elo calculation (determines how much ratings change)
        
    Returns:
        Updated Elo rating
    """
    return rating + k_factor * (actual - expected)


class EloRatingSystem:
    """
    A system for tracking and updating Elo ratings for a set of items.
    """
    
    def __init__(self, default_rating: float = 1500.0, k_factor: float = 32.0):
        """
        Initialize an Elo rating system.
        
        Args:
            default_rating: Default rating for new items
            k_factor: K-factor for Elo calculation (determines how much ratings change)
        """
        self.ratings: Dict[str, float] = {}
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.history: List[Tuple[str, str, float, float, float]] = []
        self._lock = threading.RLock()
    
    def get_rating(self, item_id: str) -> float:
        """
        Get the Elo rating for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Elo rating of the item
        """
        with self._lock:
            return self.ratings.get(item_id, self.default_rating)
    
    def record_match(self, item_a: str, item_b: str, outcome: float) -> Tuple[float, float]:
        """
        Record the outcome of a match between two items and update their Elo ratings.
        
        Args:
            item_a: ID of item A
            item_b: ID of item B
            outcome: Outcome of the match (0 for B wins, 0.5 for draw, 1 for A wins)
            
        Returns:
            Tuple of (new rating for item A, new rating for item B)
        """
        with self._lock:
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
            self.history.append((item_a, item_b, outcome, new_rating_a, new_rating_b))
            
            return new_rating_a, new_rating_b
    
    def find_closest_ratings(self, item_id: str, n: int = 5) -> List[str]:
        """
        Find the n items with the closest Elo ratings to the given item.
        
        Args:
            item_id: ID of the item
            n: Number of items to return
            
        Returns:
            List of item IDs with the closest ratings
        """
        with self._lock:
            # Get the rating for this item
            rating = self.get_rating(item_id)
            
            # Sort other items by the absolute difference in ratings
            sorted_items = sorted(
                [id for id in self.ratings.keys() if id != item_id],
                key=lambda id: abs(self.ratings[id] - rating)
            )
            
            # Return the n closest items
            return sorted_items[:n]
    
    def normalize_ratings(self, target_ratio: float = 0.5, target_threshold: float = 1.0) -> Dict[str, float]:
        """
        Normalize ratings to a distribution where a specified ratio of items have
        normalized ratings above a threshold.
        
        Args:
            target_ratio: Target ratio of items with normalized ratings above threshold
            target_threshold: Threshold for normalized ratings
            
        Returns:
            Dictionary mapping item IDs to normalized ratings
        """
        with self._lock:
            if not self.ratings:
                return {}
            
            # Get all ratings
            all_ratings = list(self.ratings.values())
            
            # Sort ratings
            all_ratings.sort()
            
            # Find the rating at the target percentile
            target_percentile = 1.0 - target_ratio
            target_index = int(target_percentile * len(all_ratings))
            target_rating = all_ratings[target_index]
            
            # Normalize ratings
            normalized = {}
            for item_id, rating in self.ratings.items():
                if rating >= target_rating:
                    # Scale ratings above the target percentile to be above the threshold
                    if max(all_ratings) == target_rating:
                        # Handle case where all ratings are the same
                        normalized_rating = target_threshold
                    else:
                        normalized_rating = target_threshold + (rating - target_rating) / (max(all_ratings) - target_rating)
                else:
                    # Scale ratings below the target percentile to be below the threshold
                    if min(all_ratings) == target_rating:
                        # Handle case where all ratings are the same
                        normalized_rating = target_threshold * 0.5
                    else:
                        normalized_rating = (rating - min(all_ratings)) / (target_rating - min(all_ratings)) * target_threshold
                
                normalized[item_id] = normalized_rating
            
            return normalized
