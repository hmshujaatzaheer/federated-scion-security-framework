"""
SCION-Specific Feature Engineering for DDoD Detection

Addresses RQ1.1: Exploiting path-aware properties for anomaly detection

SCION provides unique features not available in traditional networks:
- Path diversity (multiple paths per AS pair)
- Explicit path selection
- Hop field authentication
- Segment-based routing
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SCIONPacket:
    """SCION packet metadata"""
    src_as: int
    dst_as: int
    path_id: int
    hop_count: int
    segment_types: List[str]  # ['core', 'up', 'down']
    timestamp: float
    packet_size: int
    

class SCIONFeatureExtractor:
    """
    Extract SCION-specific features for DDoD detection
    
    Addresses RQ1.1: How to exploit path-aware properties?
    """
    
    def __init__(self):
        self.path_history = {}
        self.as_traffic = {}
        
    def extract_features(self, packet: SCIONPacket) -> np.ndarray:
        """
        Extract 20 SCION-specific features + 63 generic network features
        Total: 83 features (as per proposal)
        
        Returns:
            Feature vector (83-dimensional)
        """
        # SCION-specific features (20 features)
        scion_features = self._extract_scion_features(packet)
        
        # Generic network features (63 features) - placeholder
        generic_features = np.random.randn(63)  # Replace with real extraction
        
        # Combine
        all_features = np.concatenate([scion_features, generic_features])
        
        return all_features
    
    def _extract_scion_features(self, packet: SCIONPacket) -> np.ndarray:
        """
        20 SCION-specific features:
        
        Path-based features:
        1. Path diversity (number of available paths)
        2. Path change frequency
        3. Average hop count
        4. Path segment type distribution
        5. Path utilization ratio
        
        AS-based features:
        6. Source AS reputation score
        7. Destination AS reputation score
        8. Inter-AS traffic volume
        9. AS-level path symmetry
        10. Number of ISDs traversed
        
        Authentication features:
        11. Hop field MAC validity rate
        12. Path segment authentication success
        13. Timestamp deviation
        
        Traffic pattern features:
        14. Per-path traffic rate
        15. Path switching anomaly score
        16. Bandwidth reservation utilization
        17. Path quality metrics (latency, loss)
        
        DDoD-specific features:
        18. Multi-path traffic distribution entropy
        19. Carpet bombing indicator (many dsts, same path)
        20. Path exhaustion attack indicator
        """
        
        features = np.zeros(20)
        
        # Feature 1: Path diversity
        as_pair = (packet.src_as, packet.dst_as)
        if as_pair not in self.path_history:
            self.path_history[as_pair] = set()
        self.path_history[as_pair].add(packet.path_id)
        features[0] = len(self.path_history[as_pair])
        
        # Feature 2: Path change frequency (simplified)
        features[1] = len(self.path_history.get(as_pair, [])) / 10.0  # Normalized
        
        # Feature 3: Hop count
        features[2] = packet.hop_count
        
        # Feature 4-6: Segment type distribution
        core_segments = sum(1 for s in packet.segment_types if s == 'core')
        features[3] = core_segments / len(packet.segment_types) if packet.segment_types else 0
        
        # Feature 7-10: AS traffic patterns
        if packet.src_as not in self.as_traffic:
            self.as_traffic[packet.src_as] = {'packets': 0, 'bytes': 0}
        self.as_traffic[packet.src_as]['packets'] += 1
        self.as_traffic[packet.src_as]['bytes'] += packet.packet_size
        
        features[7] = self.as_traffic[packet.src_as]['packets']
        features[8] = self.as_traffic[packet.src_as]['bytes']
        
        # Feature 11-13: Authentication features (placeholder)
        features[10] = 1.0  # Assume valid for now
        
        # Feature 14-17: Traffic patterns
        features[13] = packet.packet_size / 1500.0  # Normalized by MTU
        
        # Feature 18-20: DDoD-specific indicators
        # Entropy of traffic distribution across paths
        if len(self.path_history.get(as_pair, [])) > 1:
            features[17] = np.log2(len(self.path_history[as_pair]))  # Simple entropy
        
        return features


# Example: Feature extraction pipeline
if __name__ == "__main__":
    extractor = SCIONFeatureExtractor()
    
    # Simulate SCION packets
    packets = [
        SCIONPacket(src_as=1, dst_as=10, path_id=1, hop_count=5, 
                   segment_types=['up', 'core', 'down'], 
                   timestamp=1.0, packet_size=1200),
        SCIONPacket(src_as=1, dst_as=10, path_id=2, hop_count=6, 
                   segment_types=['up', 'core', 'down'], 
                   timestamp=1.1, packet_size=1400),
        SCIONPacket(src_as=2, dst_as=10, path_id=3, hop_count=4, 
                   segment_types=['core', 'down'], 
                   timestamp=1.2, packet_size=800),
    ]
    
    print("SCION Feature Extraction for DDoD Detection")
    print("=" * 60)
    
    for i, packet in enumerate(packets):
        features = extractor.extract_features(packet)
        print(f"\nPacket {i+1}:")
        print(f"  Source AS: {packet.src_as}, Dest AS: {packet.dst_as}")
        print(f"  Path ID: {packet.path_id}, Hops: {packet.hop_count}")
        print(f"  Feature vector shape: {features.shape}")
        print(f"  SCION-specific features (first 20): {features[:20]}")
    
    print("\n" + "=" * 60)
    print("Features ready for federated CNN-GRU-DNN model training")
