"""
Policy network for predicting chunk importance
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import json
from pathlib import Path

from ..models import Chunk


class PolicyNetwork(nn.Module):
    """
    Predicts importance score for memory chunks.
    
    Trained using hindsight labels (whether chunk was referenced later).
    Different policy for each task type.
    """
    
    def __init__(self, task_type: str, input_dim: int = 128):
        """
        Initialize policy network.
        
        Args:
            task_type: Task type this policy is for
            input_dim: Input feature dimension
        """
        super().__init__()
        
        self.task_type = task_type
        self.input_dim = input_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Importance scorer
        self.scorer = nn.Linear(128, 1)
    
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features: Chunk features [batch_size, input_dim]
            
        Returns:
            Importance scores [batch_size, 1] in range [0, 1]
        """
        encoded = self.encoder(features)
        score = torch.sigmoid(self.scorer(encoded))
        return score
    
    def predict_importance(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        """
        Predict importance score for a chunk.
        
        Args:
            chunk: Chunk to score
            context: Optional context chunks
            
        Returns:
            Importance score in [0, 1]
        """
        self.eval()
        
        with torch.no_grad():
            # Extract features
            features = self._extract_features(chunk, context)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Predict
            score = self.forward(features_tensor)
            
            return score.item()
    
    def _extract_features(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> List[float]:
        """
        Extract features from chunk for importance prediction.
        
        Features:
        - Recency (how recent is this chunk?)
        - Chunk type (medical, code, etc.)
        - Has entities (does it contain named entities?)
        - Length (token count)
        - Position in conversation
        - Context similarity (if context provided)
        """
        from datetime import datetime
        
        features = []
        
        # Recency (normalized to [0, 1])
        # More recent = higher score
        age_seconds = (datetime.now() - chunk.timestamp).total_seconds()
        recency = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours
        features.append(recency)
        
        # Chunk type one-hot encoding
        chunk_types = ['medical', 'code', 'fact', 'preference', 'context', 'conversation', 'other']
        chunk_type_vector = [1.0 if chunk.chunk_type.value == t else 0.0 for t in chunk_types]
        features.extend(chunk_type_vector)
        
        # Length (normalized)
        length_normalized = min(chunk.tokens / 100.0, 1.0)
        features.append(length_normalized)
        
        # Has medical terms (simple heuristic)
        medical_terms = ['allergy', 'allergic', 'medication', 'diagnosis', 'symptom', 'doctor', 'patient']
        has_medical = any(term in chunk.content.lower() for term in medical_terms)
        features.append(1.0 if has_medical else 0.0)
        
        # Has code patterns
        code_patterns = ['def ', 'class ', 'import ', 'function', 'error', 'bug', '```']
        has_code = any(pattern in chunk.content.lower() for pattern in code_patterns)
        features.append(1.0 if has_code else 0.0)
        
        # Access count (normalized)
        access_normalized = min(chunk.access_count / 10.0, 1.0)
        features.append(access_normalized)
        
        # Is pinned
        features.append(1.0 if chunk.is_pinned else 0.0)
        
        # Pad or truncate to input_dim
        while len(features) < self.input_dim:
            features.append(0.0)
        features = features[:self.input_dim]
        
        return features
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'task_type': self.task_type,
            'input_dim': self.input_dim,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PolicyNetwork':
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            task_type=checkpoint['task_type'],
            input_dim=checkpoint['input_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model


def train_policy_network(
    task_type: str,
    train_data: List[Dict],
    val_data: List[Dict],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = None
) -> PolicyNetwork:
    """
    Train policy network for a specific task type.
    
    Args:
        task_type: Task type (medical, code, etc.)
        train_data: List of dicts with 'features' and 'label' (0 or 1)
        val_data: Validation data
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        save_path: Where to save model
        
    Returns:
        Trained model
    """
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    
    # Prepare data
    train_features = torch.tensor([d['features'] for d in train_data], dtype=torch.float32)
    train_labels = torch.tensor([d['label'] for d in train_data], dtype=torch.float32).unsqueeze(1)
    
    val_features = torch.tensor([d['features'] for d in val_data], dtype=torch.float32)
    val_labels = torch.tensor([d['label'] for d in val_data], dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    input_dim = train_features.shape[1]
    model = PolicyNetwork(task_type=task_type, input_dim=input_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for features, labels in pbar:
            optimizer.zero_grad()
            
            scores = model(features)
            loss = criterion(scores, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                scores = model(features)
                preds = (scores > 0.5).float()
                
                correct += (preds == labels).sum().item()
                total += len(labels)
        
        val_acc = correct / total
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}')
    
    # Save model
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f'Model saved to {save_path}')
    
    return model