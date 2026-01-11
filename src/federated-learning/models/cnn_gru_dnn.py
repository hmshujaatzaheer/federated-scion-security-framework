"""
CNN-GRU-DNN Hybrid Model for SCION DDoD Detection

Addresses RQ1.1: SCION-specific DDoD detection using deep learning

Architecture:
- CNN: Extract spatial patterns from packet sequences
- GRU: Model temporal dependencies in traffic flows  
- DNN: Final classification (benign vs attack)

Features: 83 total (20 SCION-specific + 63 generic network features)
Target Performance: 99%+ accuracy, <60s detection latency, <5% FPR
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRUDNN(nn.Module):
    """
    Hybrid CNN-GRU-DNN architecture for DDoD detection in SCION networks
    
    Args:
        input_size: Number of features (83)
        hidden_size: GRU hidden dimension (default: 128)
        num_layers: Number of GRU layers (default: 2)
        num_classes: Binary classification (default: 2)
        dropout: Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_size=83, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.3):
        super(CNNGRUDNN, self).__init__()
        
        # CNN for spatial feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        # GRU for temporal sequence modeling
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # DNN classifier
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN: (batch_size, 1, sequence_length * features)
        x = x.view(batch_size, 1, -1)
        
        # CNN layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Reshape for GRU: (batch_size, sequence_length, hidden_size)
        x = x.permute(0, 2, 1)
        
        # GRU layer
        gru_out, hidden = self.gru(x)
        
        # Take last hidden state
        x = gru_out[:, -1, :]
        
        # DNN classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class FederatedDDoDetector:
    """
    Federated DDoD Detector for SCION Networks
    
    Integrates CNN-GRU-DNN model with federated learning
    Addresses RQ1.1 and RQ1.4
    """
    
    def __init__(self, model_params=None):
        self.model = CNNGRUDNN(**(model_params or {}))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def get_model_weights(self):
        """Get model weights for federated aggregation"""
        return {name: param.data.clone() for name, param in self.model.state_dict().items()}
    
    def set_model_weights(self, weights):
        """Set model weights from federated aggregation"""
        self.model.load_state_dict(weights)


# Example usage
if __name__ == "__main__":
    print("CNN-GRU-DNN Model for SCION DDoD Detection")
    print("=" * 60)
    
    # Create model
    model = CNNGRUDNN(input_size=83)
    
    print(f"Model architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 32
    seq_length = 10
    input_features = 83
    
    dummy_input = torch.randn(batch_size, seq_length, input_features)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel ready for federated training across SCION ASes!")
