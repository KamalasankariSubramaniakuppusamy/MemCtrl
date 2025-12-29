"""
Tests for ML components
"""

import pytest
import torch

from memctrl.ml import TaskClassifier, PolicyNetwork, create_hindsight_labels
from memctrl.models import Chunk, Session, ChunkType


def test_task_classifier_init():
    """Test task classifier initialization"""
    classifier = TaskClassifier()
    
    assert classifier.num_classes == 5
    assert len(classifier.TASK_TYPES) == 5
    assert 'medical' in classifier.TASK_TYPES
    assert 'code' in classifier.TASK_TYPES


def test_task_classifier_predict():
    """Test task classifier prediction"""
    classifier = TaskClassifier()
    
    # Medical text
    medical_text = "Patient has chest pain and difficulty breathing"
    task, probs = classifier.predict(medical_text, return_probs=True)
    
    assert task in classifier.TASK_TYPES
    assert isinstance(probs, dict)
    assert len(probs) == 5
    assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)


def test_policy_network_init():
    """Test policy network initialization"""
    policy = PolicyNetwork(task_type='medical')
    
    assert policy.task_type == 'medical'
    assert policy.input_dim == 128


def test_policy_network_predict():
    """Test policy network prediction"""
    policy = PolicyNetwork(task_type='medical')
    
    chunk = Chunk(
        id='test',
        content='Patient is allergic to penicillin',
        tokens=5,
        chunk_type=ChunkType.MEDICAL
    )
    
    score = policy.predict_importance(chunk)
    
    assert 0 <= score <= 1


def test_hindsight_labeling():
    """Test hindsight label generation"""
    # Create a session with chunks
    session = Session(id='test', user_id='test')
    
    chunk1 = Chunk(id='c1', content='User has diabetes', tokens=3)
    chunk2 = Chunk(id='c2', content='Checking blood sugar', tokens=3)
    chunk3 = Chunk(id='c3', content='Diabetes management plan', tokens=3)
    
    session.add_chunk(chunk1)
    session.add_chunk(chunk2)
    session.add_chunk(chunk3)
    
    # Generate labels
    labels = create_hindsight_labels(session)
    
    assert len(labels) > 0
    assert all('features' in label for label in labels)
    assert all('label' in label for label in labels)
    assert all(label['label'] in [0.0, 1.0] for label in labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])