import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional
from pathlib import Path


class TaskClassifier(nn.Module):    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 5):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained transformer
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Label mapping
        self.label2id = {
            'medical': 0,
            'code': 1,
            'writing': 2,
            'tutoring': 3,
            'general': 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(
        self,
        text: str,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = 'cpu',
        return_probs: bool = False
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Predict task type for a conversation
        
        Args:
            text: Conversation text
            tokenizer: Tokenizer (if None, loads from model_name)
            device: Device to run on
            return_probs: If True, return probability distribution
            
        Returns:
            pred_label: Predicted task type
            probs: Optional dict of probabilities per task (if return_probs=True)
        """
        self.eval()
        
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize
        with torch.no_grad():
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get logits
            logits = self.forward(input_ids, attention_mask)
            
            # Convert to probabilities (SOFT CLASSIFICATION)
            probs = torch.softmax(logits, dim=-1)[0]
            
            # Get predicted class
            pred_idx = torch.argmax(probs).item()
            pred_label = self.id2label[pred_idx]
            
            if return_probs:
                # Return soft distribution
                prob_dict = {
                    label: probs[idx].item()
                    for idx, label in self.id2label.items()
                }
                return pred_label, prob_dict
            
            return pred_label, None
    
    def save(self, path: str, tokenizer_name: Optional[str] = None):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'tokenizer_name': tokenizer_name or self.model_name,
            'num_classes': self.num_classes,
            'label2id': self.label2id,
            'id2label': self.id2label
        }, path)
        
        print(f"✓ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TaskClassifier':
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model
        model = cls(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded from {path}")
        return model


# Convenience function
def load_task_classifier(path: str = 'models/task_classifier.pt', device: str = 'cpu') -> TaskClassifier:
    """
    Load a trained task classifier
    
    Args:
        path: Path to saved model
        device: Device to load on
        
    Returns:
        Loaded TaskClassifier
    """
    return TaskClassifier.load(path, device)