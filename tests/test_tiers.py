"""
Tests for tier management
"""

import pytest
import tempfile
from pathlib import Path

from memctrl.core.tiers import Tier0_GPU, Tier1_RAM, Tier2_Disk, TierManager
from memctrl.models import Chunk, ChunkPriority


def test_tier0_basic():
    """Test Tier 0 basic operations"""
    tier0 = Tier0_GPU(max_tokens=100)
    
    chunk = Chunk(id="c1", content="Test", tokens=10)
    
    # Add
    assert tier0.add(chunk)
    assert tier0.current_tokens == 10
    
    # Get
    retrieved = tier0.get("c1")
    assert retrieved is not None
    assert retrieved.id == "c1"
    
    # Remove
    removed = tier0.remove("c1")
    assert removed is not None
    assert tier0.current_tokens == 0


def test_tier0_capacity():
    """Test Tier 0 capacity limits"""
    tier0 = Tier0_GPU(max_tokens=100)
    
    # Add chunks until full
    for i in range(10):
        chunk = Chunk(id=f"c{i}", content=f"Test {i}", tokens=10)
        tier0.add(chunk)
    
    assert tier0.current_tokens == 100
    assert tier0.is_full()
    
    # Try to add more (should fail without force)
    extra = Chunk(id="extra", content="Extra", tokens=10)
    assert not tier0.add(extra, force=False)
    
    # With force, should evict LRU
    assert tier0.add(extra, force=True)
    assert tier0.current_tokens <= 100


def test_tier0_pinned_never_evicted():
    """Test that pinned chunks are never evicted"""
    tier0 = Tier0_GPU(max_tokens=50)
    
    # Add pinned chunk
    pinned = Chunk(id="pinned", content="Important", tokens=30, is_pinned=True)
    tier0.add(pinned, force=True)
    
    # Add normal chunks to fill
    normal = Chunk(id="normal", content="Normal", tokens=20)
    tier0.add(normal, force=True)
    
    # Try to add more - should evict normal, not pinned
    extra = Chunk(id="extra", content="Extra", tokens=10)
    tier0.add(extra, force=True)
    
    # Pinned should still be there
    assert tier0.get("pinned") is not None
    # Normal should be evicted
    assert tier0.get("normal") is None


def test_tier1_compression():
    """Test Tier 1 compression"""
    tier1 = Tier1_RAM(max_tokens=100)
    
    chunk = Chunk(id="c1", content="This is a long test content" * 10, tokens=50)
    
    # Add (should compress)
    assert tier1.add(chunk)
    
    # Should use less tokens (compressed)
    assert tier1.current_tokens < 50
    
    # Retrieve
    retrieved = tier1.get("c1")
    assert retrieved is not None
    assert retrieved.summary is not None


@pytest.fixture
def temp_tier2():
    """Create temporary Tier 2 for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    tier2 = Tier2_Disk(db_path)
    yield tier2
    
    Path(db_path).unlink(missing_ok=True)


def test_tier2_persistence(temp_tier2):
    """Test Tier 2 persistent storage"""
    chunk = Chunk(id="c1", content="Persistent", tokens=5)
    chunk.metadata['user_id'] = 'test_user'
    
    # Add
    assert temp_tier2.add(chunk)
    
    # Retrieve
    retrieved = temp_tier2.get("c1")
    assert retrieved is not None
    assert retrieved.content == "Persistent"


def test_tier2_search(temp_tier2):
    """Test Tier 2 full-text search"""
    chunks = [
        Chunk(id="c1", content="Python programming", tokens=2),
        Chunk(id="c2", content="JavaScript coding", tokens=2),
    ]
    
    for chunk in chunks:
        chunk.metadata['user_id'] = 'test_user'
        temp_tier2.add(chunk)
    
    results = temp_tier2.search("Python", user_id='test_user')
    assert len(results) == 1
    assert results[0].content == "Python programming"


def test_tier_manager_flow(temp_tier2):
    """Test TierManager promotes/demotes correctly"""
    tier0 = Tier0_GPU(max_tokens=50)
    tier1 = Tier1_RAM(max_tokens=100)
    
    manager = TierManager(tier0, tier1, temp_tier2)
    
    # Add high-priority chunk
    important = Chunk(id="imp", content="Important", tokens=10)
    important.set_importance(0.9, "medical")
    
    manager.add_chunk(important, user_id="test", session_id="s1")
    
    # Should be in Tier 0
    assert tier0.get("imp") is not None
    
    # Add low-priority chunk
    normal = Chunk(id="norm", content="Normal", tokens=10)
    normal.set_importance(0.3, "general")
    
    manager.add_chunk(normal, user_id="test", session_id="s1")
    
    # Should be in Tier 1 (low priority)
    assert tier1.get("norm") is not None
    
    # Get normal chunk (should promote)
    retrieved = manager.get_chunk("norm")
    assert retrieved is not None


def test_tier_manager_stats(temp_tier2):
    """Test TierManager statistics"""
    manager = TierManager(tier2=temp_tier2)
    
    chunk = Chunk(id="c1", content="Test", tokens=5)
    manager.add_chunk(chunk, user_id="test", session_id="s1")
    
    stats = manager.get_all_stats()
    
    assert 'tier0' in stats
    assert 'tier1' in stats
    assert 'tier2' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])