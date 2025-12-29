"""
3-Phase Training for Task Classifier
Phase 1: Real data only (anchor true patterns)
Phase 2: Weighted mixing (expand coverage)
Phase 3: Final evaluation
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from memctrl.ml.task_classifier import TaskClassifier


class TaskDataset(Dataset):
    """Dataset with source tracking"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.label2id = {
            'medical': 0,
            'code': 1,
            'writing': 2,
            'tutoring': 3,
            'general': 4
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.label2id[item['label']]),
            'is_synthetic': item.get('is_synthetic', False)
        }


def load_data(data_file='data/task_classifier_data.json'):
    """Load training data"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data['train'], data['val'], data['test']


def filter_real_only(data):
    """Filter to only real data"""
    return [item for item in data if not item.get('is_synthetic', False)]


def compute_sample_weights(data, synthetic_weight=0.3):
    """
    Compute sample weights
    Real: 1.0, Synthetic: synthetic_weight
    """
    weights = []
    for item in data:
        if item.get('is_synthetic', False):
            weights.append(synthetic_weight)
        else:
            weights.append(1.0)
    return weights


def evaluate(model, dataloader, device, split_name='Val'):
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_is_synthetic = []
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {split_name}', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_is_synthetic.extend(batch['is_synthetic'].numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Overall accuracy
    overall_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # Real-only accuracy
    real_mask = ~np.array(all_is_synthetic)
    real_acc = (np.array(all_preds)[real_mask] == np.array(all_labels)[real_mask]).mean() if real_mask.sum() > 0 else 0.0
    
    # Synthetic-only accuracy
    synthetic_mask = np.array(all_is_synthetic)
    synthetic_acc = (np.array(all_preds)[synthetic_mask] == np.array(all_labels)[synthetic_mask]).mean() if synthetic_mask.sum() > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'overall_acc': overall_acc,
        'real_acc': real_acc,
        'synthetic_acc': synthetic_acc,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def train_phase1_real_only(model, train_data, val_data, tokenizer, device, epochs=3, lr=2e-5):
    """PHASE 1: Train on REAL data only"""
    print("\n" + "="*70)
    print("PHASE 1: Training on REAL data only")
    print("="*70)
    
    train_real = filter_real_only(train_data)
    val_real = filter_real_only(val_data)
    
    print(f"Train: {len(train_real)} real examples")
    print(f"Val:   {len(val_real)} real examples")
    
    train_dataset = TaskDataset(train_real, tokenizer)
    val_dataset = TaskDataset(val_real, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Phase 1 - Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        val_results = evaluate(model, val_loader, device, 'Val (Real)')
        
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_results['loss']:.4f}, Val Acc={val_results['overall_acc']:.4f}")
    
    print("\n✓ Phase 1 complete: Model anchored on real data")
    return model


def train_phase2_weighted_mixing(model, train_data, val_data, tokenizer, device, epochs=3, lr=1e-5, synthetic_weight=0.3):
    """PHASE 2: Weighted mixing (Real + Synthetic)"""
    print("\n" + "="*70)
    print("PHASE 2: Weighted mixing (Real + Synthetic)")
    print(f"Synthetic weight: {synthetic_weight}")
    print("="*70)
    
    real_count = sum(1 for item in train_data if not item.get('is_synthetic', False))
    synthetic_count = len(train_data) - real_count
    
    print(f"Train: {real_count} real + {synthetic_count} synthetic = {len(train_data)} total")
    
    train_dataset = TaskDataset(train_data, tokenizer)
    val_dataset = TaskDataset(val_data, tokenizer)
    
    # Weighted sampling
    sample_weights = compute_sample_weights(train_data, synthetic_weight)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Phase 2 - Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        val_results = evaluate(model, val_loader, device, 'Val (Mixed)')
        
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Overall={val_results['overall_acc']:.4f}, Val Real={val_results['real_acc']:.4f}, Val Synthetic={val_results['synthetic_acc']:.4f}")
    
    print("\n✓ Phase 2 complete: Coverage expanded")
    return model


def train_phase3_final_validation(model, test_data, tokenizer, device):
    """PHASE 3: Final test evaluation"""
    print("\n" + "="*70)
    print("PHASE 3: Final Test Set Evaluation")
    print("="*70)
    
    test_dataset = TaskDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    results = evaluate(model, test_loader, device, 'Test')
    
    print("\nFINAL RESULTS:")
    print(f"  Test Loss:      {results['loss']:.4f}")
    print(f"  Test Overall:   {results['overall_acc']:.4f}")
    print(f"  Test Real:      {results['real_acc']:.4f}")
    print(f"  Test Synthetic: {results['synthetic_acc']:.4f}")
    
    # Classification report
    label_names = ['medical', 'code', 'writing', 'tutoring', 'general']
    print("\nPer-class Performance:")
    print(classification_report(results['all_labels'], results['all_preds'], target_names=label_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(results['all_labels'], results['all_preds'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=150)
    print("\n✓ Confusion matrix saved to results/confusion_matrix.png")
    
    return results


def main():
    """Main training pipeline"""
    print("="*70)
    print("3-PHASE TASK CLASSIFIER TRAINING")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, test_data = load_data()
    
    train_real = sum(1 for item in train_data if not item.get('is_synthetic', False))
    train_synthetic = len(train_data) - train_real
    
    print(f"\nDataset composition:")
    print(f"  Train: {train_real} real, {train_synthetic} synthetic ({train_synthetic/len(train_data)*100:.1f}% synthetic)")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Initialize
    print("\nInitializing model...")
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TaskClassifier(model_name=model_name, num_classes=5)
    model.to(device)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # PHASE 1
    model = train_phase1_real_only(model, train_data, val_data, tokenizer, device, epochs=3, lr=2e-5)
    
    # PHASE 2
    model = train_phase2_weighted_mixing(model, train_data, val_data, tokenizer, device, epochs=3, lr=1e-5, synthetic_weight=0.3)
    
    # PHASE 3
    results = train_phase3_final_validation(model, test_data, tokenizer, device)
    
    # Save
    save_path = 'models/task_classifier.pt'
    model.save(save_path, tokenizer_name=model_name)
    
    print(f"\n Training complete!")
    print(f"Final test accuracy (real): {results['real_acc']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()