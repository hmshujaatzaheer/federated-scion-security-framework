"""
Federated Averaging (FedAvg) Protocol for SCION DDoD Detection

Implementation of federated learning protocol exploiting SCION path-aware properties.
Addresses RQ1.1: How can federated protocols exploit SCION path-aware properties?

Reference:
    McMahan et al. "Communication-Efficient Learning of Deep Networks 
    from Decentralized Data" (AISTATS 2017)
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedAvgClient:
    """
    Federated Learning Client for SCION AS
    
    Each SCION AS runs a local client that:
    1. Trains on local traffic data
    2. Computes model updates
    3. Shares updates (not raw data) for privacy
    """
    
    def __init__(self, client_id: int, learning_rate: float = 0.01):
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.model_weights = None
        self.local_data_size = 0
        
    def train_local_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         epochs: int = 5) -> Dict[str, np.ndarray]:
        """
        Train model on local SCION traffic data
        
        Args:
            X_train: Local network traffic features (SCION-specific + generic)
            y_train: Attack labels (0=benign, 1=DDoD attack)
            epochs: Number of local training epochs
            
        Returns:
            Updated model weights
        """
        self.local_data_size = len(X_train)
        
        # Initialize weights if first time
        if self.model_weights is None:
            feature_dim = X_train.shape[1]
            self.model_weights = {
                'W': np.random.randn(feature_dim, 1) * 0.01,
                'b': np.zeros((1, 1))
            }
        
        # Local training loop (simple logistic regression for demonstration)
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X_train, self.model_weights['W']) + self.model_weights['b']
            predictions = self._sigmoid(z)
            
            # Compute loss
            loss = self._binary_cross_entropy(y_train, predictions)
            
            # Backward pass
            m = X_train.shape[0]
            dW = (1/m) * np.dot(X_train.T, (predictions - y_train))
            db = (1/m) * np.sum(predictions - y_train)
            
            # Update weights
            self.model_weights['W'] -= self.learning_rate * dW
            self.model_weights['b'] -= self.learning_rate * db
            
            if epoch % 2 == 0:
                logger.info(f"Client {self.client_id} - Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        
        return self.model_weights
    
    def get_model_update(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Get model update and data size for federated aggregation
        
        Returns:
            (model_weights, local_data_size)
        """
        return self.model_weights, self.local_data_size
    
    def set_global_model(self, global_weights: Dict[str, np.ndarray]):
        """Update local model with global aggregated weights"""
        self.model_weights = global_weights
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class FedAvgServer:
    """
    Federated Averaging Server (can be decentralized via smart contract)
    
    Aggregates model updates from multiple SCION ASes
    Addresses RQ1.1: Federated DDoD detection across path-aware network
    """
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.global_model = None
        self.round = 0
        
    def aggregate_models(self, client_updates: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, np.ndarray]:
        """
        Federated Averaging: Weighted average by dataset size
        
        Args:
            client_updates: List of (model_weights, data_size) from clients
            
        Returns:
            Aggregated global model weights
        """
        total_samples = sum(data_size for _, data_size in client_updates)
        
        # Initialize aggregated weights
        aggregated = None
        
        for weights, data_size in client_updates:
            weight_factor = data_size / total_samples
            
            if aggregated is None:
                aggregated = {
                    key: weight_factor * value 
                    for key, value in weights.items()
                }
            else:
                for key in weights.keys():
                    aggregated[key] += weight_factor * weights[key]
        
        self.global_model = aggregated
        self.round += 1
        
        logger.info(f"Round {self.round}: Aggregated models from {len(client_updates)} clients")
        
        return self.global_model


def federated_learning_simulation(num_clients: int = 5, num_rounds: int = 10):
    """
    Simulate federated learning across SCION ASes
    
    Args:
        num_clients: Number of SCION ASes participating
        num_rounds: Number of federated learning rounds
    """
    # Create server
    server = FedAvgServer(num_clients)
    
    # Create clients (representing SCION ASes)
    clients = [FedAvgClient(i) for i in range(num_clients)]
    
    # Simulate training
    for round_num in range(num_rounds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Federated Learning Round {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*50}")
        
        client_updates = []
        
        # Each client trains locally
        for client in clients:
            # Generate synthetic SCION traffic data (replace with real data)
            X_train = np.random.randn(100, 20)  # 100 samples, 20 features
            y_train = np.random.randint(0, 2, (100, 1))  # Binary labels
            
            # Local training
            client.train_local_model(X_train, y_train, epochs=3)
            
            # Get model update
            client_updates.append(client.get_model_update())
        
        # Server aggregates
        global_model = server.aggregate_models(client_updates)
        
        # Distribute global model back to clients
        for client in clients:
            client.set_global_model(global_model)
    
    logger.info(f"\n{'='*50}")
    logger.info("Federated Learning Complete!")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    # Run federated learning simulation
    federated_learning_simulation(num_clients=5, num_rounds=10)
