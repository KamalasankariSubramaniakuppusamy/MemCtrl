"""
Memory tier implementations for MemCtrl
"""

from typing import List, Optional, Dict
from datetime import datetime
from collections import OrderedDict

from ..models import Chunk, ChunkPriority
from ..config import get_config
from ..storage.sqlite_store import SQLiteStore


class Tier0_GPU:
    """
    Tier 0: Active GPU memory
    - Stores most important/recent chunks
    - Fast access
    - Limited capacity (e.g., 4GB / 4096 tokens)
    """
    
    def __init__(self, max_tokens: Optional[int] = None):
        """
        Initialize Tier 0.
        
        Args:
            max_tokens: Maximum tokens to store. If None, uses config.
        """
        config = get_config()
        self.max_tokens = max_tokens or config.get_tier0_tokens()
        self.current_tokens = 0
        
        # OrderedDict maintains insertion order for LRU
        self.storage: OrderedDict[str, Chunk] = OrderedDict()
    
    def add(self, chunk: Chunk, force: bool = False) -> bool:
        """
        Add chunk to Tier 0.
        
        Args:
            chunk: Chunk to add
            force: If True, add even if full (evict LRU first)
            
        Returns:
            True if added, False if full and force=False
        """
        # If already exists, update it
        if chunk.id in self.storage:
            self.remove(chunk.id)
        
        # Check if we have space
        if not force and self.current_tokens + chunk.tokens > self.max_tokens:
            return False
        
        # Make space if needed
        while self.current_tokens + chunk.tokens > self.max_tokens:
            if not self._evict_lru():
                return False  # Can't evict (all pinned?)
        
        # Add chunk
        self.storage[chunk.id] = chunk
        self.current_tokens += chunk.tokens
        chunk.update_access()
        
        return True
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID and update access time"""
        chunk = self.storage.get(chunk_id)
        if chunk:
            chunk.update_access()
            # Move to end (most recently used)
            self.storage.move_to_end(chunk_id)
        return chunk
    
    def remove(self, chunk_id: str) -> Optional[Chunk]:
        """Remove and return chunk"""
        chunk = self.storage.pop(chunk_id, None)
        if chunk:
            self.current_tokens -= chunk.tokens
        return chunk
    
    def get_all(self) -> List[Chunk]:
        """Get all chunks"""
        return list(self.storage.values())
    
    def get_pinned(self) -> List[Chunk]:
        """Get all user-pinned chunks"""
        return [c for c in self.storage.values() if c.is_pinned]
    
    def get_recent(self, n: int = 10) -> List[Chunk]:
        """Get n most recent chunks"""
        chunks = sorted(
            self.storage.values(),
            key=lambda c: c.timestamp,
            reverse=True
        )
        return chunks[:n]
    
    def is_full(self) -> bool:
        """Check if tier is at capacity"""
        config = get_config()
        return self.current_tokens >= self.max_tokens * config.eviction_threshold
    
    def get_usage(self) -> Dict[str, float]:
        """Get usage statistics"""
        return {
            'current_tokens': self.current_tokens,
            'max_tokens': self.max_tokens,
            'utilization': self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0,
            'num_chunks': len(self.storage),
        }
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used chunk that's not pinned.
        
        Returns:
            True if evicted, False if nothing to evict
        """
        # Iterate from oldest to newest (OrderedDict maintains order)
        for chunk_id, chunk in list(self.storage.items()):
            # Don't evict pinned chunks
            if chunk.priority == ChunkPriority.USER_PINNED:
                continue
            
            # Evict this chunk
            self.remove(chunk_id)
            return True
        
        # All chunks are pinned
        return False
    
    def clear(self):
        """Clear all chunks (for testing)"""
        self.storage.clear()
        self.current_tokens = 0


class Tier1_RAM:
    """
    Tier 1: Compressed RAM storage
    - Stores compressed summaries
    - Medium access speed
    - Larger capacity than Tier 0 (e.g., 16K tokens after compression)
    """
    
    def __init__(self, max_tokens: Optional[int] = None):
        """
        Initialize Tier 1.
        
        Args:
            max_tokens: Maximum tokens to store (compressed). If None, uses config.
        """
        config = get_config()
        self.max_tokens = max_tokens or config.get_tier1_tokens()
        self.current_tokens = 0
        
        # Store compressed chunks
        self.storage: Dict[str, Chunk] = {}
    
    def add(self, chunk: Chunk, compressed: bool = False) -> bool:
        """
        Add chunk to Tier 1.
        
        Args:
            chunk: Chunk to add
            compressed: If True, chunk is already compressed
            
        Returns:
            True if added, False if full
        """
        # Compress if needed
        if not compressed and not chunk.summary:
            self._compress(chunk)
        
        # Check space (use compressed size)
        compressed_tokens = self._get_compressed_tokens(chunk)
        
        if self.current_tokens + compressed_tokens > self.max_tokens:
            return False
        
        # Add
        self.storage[chunk.id] = chunk
        self.current_tokens += compressed_tokens
        
        return True
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.storage.get(chunk_id)
    
    def remove(self, chunk_id: str) -> Optional[Chunk]:
        """Remove chunk"""
        chunk = self.storage.pop(chunk_id, None)
        if chunk:
            compressed_tokens = self._get_compressed_tokens(chunk)
            self.current_tokens -= compressed_tokens
        return chunk
    
    def get_all(self) -> List[Chunk]:
        """Get all chunks"""
        return list(self.storage.values())
    
    def decompress(self, chunk: Chunk) -> Chunk:
        """
        Decompress chunk (restore full content).
        For now, just return the chunk with summary as content.
        Real implementation would use LLM to expand summary.
        """
        # TODO: Implement real decompression with LLM
        if chunk.summary and not chunk.content:
            chunk.content = chunk.summary
        return chunk
    
    def get_usage(self) -> Dict[str, float]:
        """Get usage statistics"""
        return {
            'current_tokens': self.current_tokens,
            'max_tokens': self.max_tokens,
            'utilization': self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0,
            'num_chunks': len(self.storage),
        }
    
    def _compress(self, chunk: Chunk):
        """
        Compress chunk in-place.
        For now, use simple truncation as summary.
        Real implementation would use LLM summarization.
        """
        config = get_config()
        
        if not chunk.summary:
            # Simple compression: truncate to summary
            max_summary_tokens = int(chunk.tokens / config.compression_ratio)
            words = chunk.content.split()
            summary_words = words[:max_summary_tokens * 2]  # Rough estimate
            chunk.summary = ' '.join(summary_words) + '...'
            chunk.compression_ratio = config.compression_ratio
    
    def _get_compressed_tokens(self, chunk: Chunk) -> int:
        """Get token count after compression"""
        if chunk.summary:
            # Estimate: summary is 1/compression_ratio of original
            config = get_config()
            return int(chunk.tokens / config.compression_ratio)
        return chunk.tokens
    
    def clear(self):
        """Clear all chunks"""
        self.storage.clear()
        self.current_tokens = 0


class Tier2_Disk:
    """
    Tier 2: Persistent disk storage
    - Uses SQLite for storage
    - Unlimited capacity
    - Slowest access (but still fast for SQLite)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Tier 2.
        
        Args:
            db_path: Path to SQLite database. If None, uses config.
        """
        self.store = SQLiteStore(db_path)
    
    def add(self, chunk: Chunk) -> bool:
        """Add chunk to persistent storage"""
        return self.store.store_chunk(chunk)
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.store.retrieve_chunk(chunk_id)
    
    def remove(self, chunk_id: str) -> bool:
        """Remove chunk"""
        return self.store.delete_chunk(chunk_id)
    
    def search(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Chunk]:
        """Full-text search"""
        return self.store.search_chunks(query, user_id, limit)
    
    def get_pinned(self, user_id: str) -> List[Chunk]:
        """Get all pinned chunks for user"""
        return self.store.get_pinned_chunks(user_id)
    
    def get_by_session(self, session_id: str) -> List[Chunk]:
        """Get all chunks in a session"""
        return self.store.get_chunks_by_session(session_id)
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get storage statistics"""
        return self.store.get_stats(user_id)


class TierManager:
    """
    Manages all three tiers and handles promotion/demotion.
    """
    
    def __init__(self, tier0: Optional[Tier0_GPU] = None,
                 tier1: Optional[Tier1_RAM] = None,
                 tier2: Optional[Tier2_Disk] = None):
        """
        Initialize tier manager.
        
        Args:
            tier0: Tier 0 instance (or creates new)
            tier1: Tier 1 instance (or creates new)
            tier2: Tier 2 instance (or creates new)
        """
        self.tier0 = tier0 or Tier0_GPU()
        self.tier1 = tier1 or Tier1_RAM()
        self.tier2 = tier2 or Tier2_Disk()
    
    def add_chunk(self, chunk: Chunk, user_id: str, session_id: str) -> bool:
        """
        Add chunk to appropriate tier.
        Strategy: Try Tier 0 first, fall back to Tier 1, always persist to Tier 2.
        """
        # Add metadata
        chunk.metadata['user_id'] = user_id
        chunk.metadata['session_id'] = session_id
        
        # Always persist to Tier 2
        self.tier2.add(chunk)
        
        # Try to add to Tier 0 (active memory)
        if chunk.is_pinned or chunk.get_priority_value() > 50:
            if self.tier0.add(chunk, force=chunk.is_pinned):
                return True
        
        # Fall back to Tier 1 (compressed)
        return self.tier1.add(chunk)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get chunk from any tier.
        Promotes to higher tier on access.
        """
        # Try Tier 0 first (fastest)
        chunk = self.tier0.get(chunk_id)
        if chunk:
            return chunk
        
        # Try Tier 1
        chunk = self.tier1.get(chunk_id)
        if chunk:
            # Promote to Tier 0 if possible
            self.promote_to_tier0(chunk)
            return chunk
        
        # Try Tier 2
        chunk = self.tier2.get(chunk_id)
        if chunk:
            # Promote to Tier 1 or Tier 0
            if chunk.is_pinned or chunk.get_priority_value() > 50:
                self.promote_to_tier0(chunk)
            else:
                self.promote_to_tier1(chunk)
            return chunk
        
        return None
    
    def remove_chunk(self, chunk_id: str):
        """Remove chunk from all tiers"""
        self.tier0.remove(chunk_id)
        self.tier1.remove(chunk_id)
        self.tier2.remove(chunk_id)
    
    def promote_to_tier0(self, chunk: Chunk) -> bool:
        """Promote chunk to Tier 0"""
        if self.tier0.add(chunk, force=chunk.is_pinned):
            self.tier1.remove(chunk.id)
            return True
        return False
    
    def promote_to_tier1(self, chunk: Chunk) -> bool:
        """Promote chunk to Tier 1"""
        if self.tier1.add(chunk):
            return True
        return False
    
    def demote_to_tier1(self, chunk_id: str) -> bool:
        """Demote chunk from Tier 0 to Tier 1"""
        chunk = self.tier0.remove(chunk_id)
        if chunk:
            return self.tier1.add(chunk)
        return False
    
    def demote_to_tier2(self, chunk_id: str):
        """Demote chunk to Tier 2 only"""
        self.tier0.remove(chunk_id)
        self.tier1.remove(chunk_id)
        # Already in Tier 2
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all tiers"""
        return {
            'tier0': self.tier0.get_usage(),
            'tier1': self.tier1.get_usage(),
            'tier2': self.tier2.get_stats(),
        }