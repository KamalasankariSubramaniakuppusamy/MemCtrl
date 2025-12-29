"""
Tests for Task Classifier
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from memctrl.ml.task_classifier import TaskClassifier
from transformers import AutoTokenizer


def test_task_classifier_init():
    """Test classifier initialization"""
    model = TaskClassifier(model_name='distilbert-base-uncased', num_classes=5)
    
    assert model.num_classes == 5
    assert model.model_name == 'distilbert-base-uncased'
    assert len(model.label2id) == 5
    assert 'medical' in model.label2id
    assert 'code' in model.label2id


def test_task_classifier_forward():
    """Test forward pass"""
    model = TaskClassifier()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    text = "I have chest pain and fever"
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    
    assert logits.shape == (1, 5)  # batch_size=1, num_classes=5


def test_task_classifier_predict():
    """Test prediction"""
    model = TaskClassifier()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    text = "I have chest pain and fever"
    pred_label, probs = model.predict(text, tokenizer, return_probs=True)
    
    assert pred_label in ['medical', 'code', 'writing', 'tutoring', 'general']
    assert isinstance(probs, dict)
    assert len(probs) == 5
    assert all(0 <= v <= 1 for v in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 0.01  # Probabilities sum to 1


def test_task_classifier_save_load(tmp_path):
    """Test save and load"""
    model = TaskClassifier()
    
    # Save
    save_path = tmp_path / "test_model.pt"
    model.save(str(save_path))
    
    assert save_path.exists()
    
    # Load
    loaded_model = TaskClassifier.load(str(save_path))
    
    assert loaded_model.num_classes == model.num_classes
    assert loaded_model.model_name == model.model_name


if __name__ == "__main__":
    pytest.main([__file__, '-v'])