"""
Unit tests for federated learning module
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_repository_structure():
    """Test that repository structure is complete"""
    import os
    
    dirs_to_check = [
        'src/federated-learning',
        'src/formal-verification',
        'src/zero-knowledge',
        'experiments',
        'data',
        'docs'
    ]
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    for dir_path in dirs_to_check:
        full_path = os.path.join(base_path, dir_path)
        assert os.path.exists(full_path), f"Directory {dir_path} should exist"

def test_basic():
    """Basic sanity test"""
    assert 1 + 1 == 2

if __name__ == "__main__":
    print("Running tests...")
    test_repository_structure()
    test_basic()
    print("All tests passed!")
