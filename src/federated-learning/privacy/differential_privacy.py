"""
Differential Privacy for Federated Learning

Addresses RQ1.3: Privacy-preserving federated learning
Implements (ε, δ)-differential privacy via Gaussian noise

Reference:
    Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for federated learning
    
    Adds calibrated Gaussian noise to model updates to protect
    individual client data privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (smaller = more privacy)
            delta: Failure probability
            sensitivity: L2 sensitivity of model updates
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Compute noise scale using Gaussian mechanism
        self.noise_scale = self._compute_noise_scale()
        
        logger.info(f"DP initialized: (ε={epsilon}, δ={delta}), noise_scale={self.noise_scale:.4f}")
    
    def _compute_noise_scale(self) -> float:
        """
        Compute Gaussian noise scale for (ε, δ)-DP
        
        σ = (Δf / ε) * √(2 ln(1.25/δ))
        where Δf is the L2 sensitivity
        """
        return (self.sensitivity / self.epsilon) * np.sqrt(2 * np.log(1.25 / self.delta))
    
    def add_noise_to_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add Gaussian noise to gradients for differential privacy
        
        Args:
            gradients: Dictionary of gradient arrays
            
        Returns:
            Noisy gradients preserving (ε, δ)-DP
        """
        noisy_gradients = {}
        
        for key, grad in gradients.items():
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_scale, grad.shape)
            noisy_gradients[key] = grad + noise
            
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict[str, np.ndarray], 
                       clip_norm: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Clip gradients to bound sensitivity
        
        Args:
            gradients: Original gradients
            clip_norm: Maximum L2 norm
            
        Returns:
            Clipped gradients
        """
        clipped = {}
        
        for key, grad in gradients.items():
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm > clip_norm:
                clipped[key] = grad * (clip_norm / grad_norm)
            else:
                clipped[key] = grad
        
        return clipped


class PrivateFedAvgClient:
    """
    Federated Averaging client with differential privacy
    
    Combines FedAvg with DP to protect client data privacy
    Addresses RQ1.3 and RQ1.4 (privacy-performance trade-off)
    """
    
    def __init__(self, client_id: int, epsilon: float = 1.0, delta: float = 1e-5):
        self.client_id = client_id
        self.dp = DifferentialPrivacy(epsilon=epsilon, delta=delta)
        self.model_weights = None
        
    def train_with_privacy(self, X_train: np.ndarray, y_train: np.ndarray, 
                          epochs: int = 5, clip_norm: float = 1.0):
        """
        Train model with differential privacy guarantees
        """
        # Initialize model
        if self.model_weights is None:
            feature_dim = X_train.shape[1]
            self.model_weights = {
                'W': np.random.randn(feature_dim, 1) * 0.01,
                'b': np.zeros((1, 1))
            }
        
        for epoch in range(epochs):
            # Compute gradients
            gradients = self._compute_gradients(X_train, y_train)
            
            # Clip gradients (bound sensitivity)
            clipped_gradients = self.dp.clip_gradients(gradients, clip_norm)
            
            # Add noise for privacy
            noisy_gradients = self.dp.add_noise_to_gradients(clipped_gradients)
            
            # Update model
            learning_rate = 0.01
            for key in self.model_weights.keys():
                self.model_weights[key] -= learning_rate * noisy_gradients[key]
        
        logger.info(f"Client {self.client_id}: Private training complete (ε={self.dp.epsilon})")
        
        return self.model_weights
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients (simplified)"""
        z = np.dot(X, self.model_weights['W']) + self.model_weights['b']
        predictions = 1 / (1 + np.exp(-z))
        
        m = X.shape[0]
        dW = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y, axis=0, keepdims=True)
        
        return {'W': dW, 'b': db}


# Example: Privacy-utility trade-off analysis (RQ1.4)
if __name__ == "__main__":
    print("Differential Privacy Trade-off Analysis")
    print("=" * 50)
    
    # Test different epsilon values
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for eps in epsilon_values:
        dp = DifferentialPrivacy(epsilon=eps, delta=1e-5)
        
        # Simulate gradient
        grad = np.array([1.0, 2.0, 3.0])
        noisy_grad = dp.add_noise_to_gradients({'grad': grad})['grad']
        
        noise_magnitude = np.linalg.norm(noisy_grad - grad)
        
        print(f"ε={eps:.1f}: Noise magnitude = {noise_magnitude:.4f}")
    
    print("\nConclusion: Lower ε → More privacy → More noise → Lower utility")
