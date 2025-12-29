"""
Train task classifier on prepared data
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple

from memctrl.ml import train_task_classifier


def load_prepared_data(data_file: str = "data/task_classifier_data.json") -> Tuple[List, List, List]:
    """Load prepared training data"""
    print(f"Loading data from {data_file}...")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    train_data = [(item['text'], item['label']) for item in data['train']]
    val_data = [(item['text'], item['label']) for item in data['val']]
    test_data = [(item['text'], item['label']) for item in data['test']]
    
    print(f"✓ Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test examples")
    
    return train_data, val_data, test_data


def main():
    """Main training function"""
    print("="*60)
    print("Training Task Classifier")
    print("="*60)
    
    # Load data
    train_data, val_data, test_data = load_prepared_data()
    
    # Train model
    print("\nStarting training...")
    print("This will take 10-30 minutes depending on your hardware.\n")
    
    model = train_task_classifier(
        train_data=train_data,
        val_data=val_data,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        save_path='models/task_classifier.pt'
    )
    
    # Test
    print("\n" + "="*60)
    print("Testing on held-out test set...")
    
    correct = 0
    total = 0
    
    for text, true_label in test_data[:100]:  # Test on subset
        pred_label = model.predict(text)
        if pred_label == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("="*60)
    
    print("\nTraining complete!")
    print("Model saved to: models/task_classifier.pt")


if __name__ == "__main__":
    main()