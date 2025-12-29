"""
Create hindsight labels for policy network training
"""

from typing import List, Dict, Tuple
from ..models import Chunk, Session


def create_hindsight_labels(session: Session) -> List[Dict]:
    """
    Create hindsight labels from a completed conversation.
    
    For each chunk, determine if it was "important" based on whether
    it was referenced or needed in later parts of the conversation.
    
    Args:
        session: Completed conversation session
        
    Returns:
        List of training examples with features and labels
    """
    chunks = session.chunks
    labels = []
    
    for i, chunk in enumerate(chunks):
        # Skip last few chunks (no future to check)
        if i >= len(chunks) - 2:
            continue
        
        # Check if this chunk was important
        was_important = check_if_important(chunk, chunks[i+1:])
        
        # Extract features (using PolicyNetwork's feature extraction)
        from .policy_network import PolicyNetwork
        dummy_policy = PolicyNetwork(task_type='general')
        features = dummy_policy._extract_features(chunk, chunks[:i])
        
        labels.append({
            'chunk_id': chunk.id,
            'features': features,
            'label': 1.0 if was_important else 0.0,
            'task_type': session.task_type or 'general'
        })
    
    return labels


def check_if_important(chunk: Chunk, future_chunks: List[Chunk]) -> bool:
    """
    Check if a chunk was important based on future references.
    
    Heuristics:
    - Was explicitly referenced (coreference)
    - Contains entities mentioned later
    - Contains keywords that appear frequently later
    - Is safety-critical (medical, security)
    """
    # Always important if pinned
    if chunk.is_pinned:
        return True
    
    # Safety-critical chunks are important
    if chunk.chunk_type.value in ['medical']:
        return True
    
    # Check for entity/keyword overlap with future
    chunk_words = set(chunk.content.lower().split())
    
    # Extract key terms (simple: words > 4 chars)
    key_terms = {w for w in chunk_words if len(w) > 4}
    
    if not key_terms:
        return False
    
    # Check overlap with future chunks
    overlap_count = 0
    for future_chunk in future_chunks:
        future_words = set(future_chunk.content.lower().split())
        overlap = key_terms & future_words
        
        if len(overlap) >= 2:  # At least 2 shared key terms
            overlap_count += 1
    
    # Important if referenced in multiple future chunks
    return overlap_count >= 2


def generate_training_data_from_dataset(
    dataset_name: str,
    num_examples: int = 1000
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data from a dataset.
    
    This is a placeholder - actual implementation would:
    1. Load conversations from dataset
    2. Split into sessions
    3. Generate hindsight labels
    4. Split into train/val
    
    Args:
        dataset_name: Name of dataset (e.g., 'ubuntu', 'meddialog')
        num_examples: Number of examples to generate
        
    Returns:
        (train_data, val_data) tuple
    """
    # TODO: Implement actual dataset loading
    # For now, return dummy data
    
    import random
    
    def generate_dummy_example():
        # Random features
        features = [random.random() for _ in range(128)]
        # Random label
        label = random.choice([0.0, 1.0])
        
        return {'features': features, 'label': label}
    
    all_data = [generate_dummy_example() for _ in range(num_examples)]
    
    # 80/20 split
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    return train_data, val_data