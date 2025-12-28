"""
Krum: Byzantine-Robust Federated Aggregation

Addresses RQ1.2: Byzantine robustness in federated learning
Resists up to f < n/2 - 1 malicious clients

Reference:
    Blanchard et al. "Machine Learning with Adversaries: 
    Byzantine Tolerant Gradient Descent" (NeurIPS 2017)
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class KrumAggregator:
    """
    Krum aggregation for Byzantine-robust federated learning
    
    Selects the most representative model update by computing
    distances to other updates, excluding outliers.
    """
    
    def __init__(self, num_byzantine: int):
        """
        Args:
            num_byzantine: Maximum number of Byzantine (malicious) clients
        """
        self.num_byzantine = num_byzantine
        
    def aggregate(self, model_updates: List[np.ndarray]) -> np.ndarray:
        """
        Krum aggregation: Select model closest to cluster
        
        Args:
            model_updates: List of model weight arrays from clients
            
        Returns:
            Selected robust model update
        """
        n = len(model_updates)
        f = self.num_byzantine
        
        # Number of closest clients to consider
        m = n - f - 2
        
        # Compute pairwise distances
        scores = []
        for i, update_i in enumerate(model_updates):
            distances = []
            for j, update_j in enumerate(model_updates):
                if i != j:
                    dist = np.linalg.norm(update_i - update_j)
                    distances.append(dist)
            
            # Sort and take m closest
            distances.sort()
            score = sum(distances[:m])
            scores.append(score)
        
        # Select update with minimum score (most representative)
        selected_idx = np.argmin(scores)
        
        logger.info(f"Krum selected client {selected_idx} (score: {scores[selected_idx]:.4f})")
        
        return model_updates[selected_idx]


class MultiKrumAggregator:
    """
    Multi-Krum: Average of top-k Krum selections
    More robust than single Krum
    """
    
    def __init__(self, num_byzantine: int, k: int = 3):
        self.num_byzantine = num_byzantine
        self.k = k  # Number of models to average
        
    def aggregate(self, model_updates: List[np.ndarray]) -> np.ndarray:
        """
        Multi-Krum: Average k models with lowest Krum scores
        """
        n = len(model_updates)
        f = self.num_byzantine
        m = n - f - 2
        
        # Compute Krum scores
        scores = []
        for i, update_i in enumerate(model_updates):
            distances = []
            for j, update_j in enumerate(model_updates):
                if i != j:
                    dist = np.linalg.norm(update_i - update_j)
                    distances.append(dist)
            distances.sort()
            score = sum(distances[:m])
            scores.append((score, i))
        
        # Sort by score and select top k
        scores.sort()
        selected_indices = [idx for _, idx in scores[:self.k]]
        
        # Average selected models
        selected_models = [model_updates[i] for i in selected_indices]
        aggregated = np.mean(selected_models, axis=0)
        
        logger.info(f"Multi-Krum selected clients {selected_indices}")
        
        return aggregated


# Example usage
if __name__ == "__main__":
    # Simulate 10 clients, 2 are Byzantine
    num_clients = 10
    num_byzantine = 2
    
    # Generate model updates (100-dim weight vectors)
    honest_updates = [np.random.randn(100) for _ in range(num_clients - num_byzantine)]
    
    # Byzantine clients send malicious updates (extreme values)
    byzantine_updates = [np.random.randn(100) * 100 for _ in range(num_byzantine)]
    
    all_updates = honest_updates + byzantine_updates
    
    # Apply Krum
    krum = KrumAggregator(num_byzantine=num_byzantine)
    robust_model = krum.aggregate(all_updates)
    
    print(f"Krum aggregation complete. Selected model norm: {np.linalg.norm(robust_model):.4f}")
