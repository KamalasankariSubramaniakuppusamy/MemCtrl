"""
User: Represents a user and their memory state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Dict, Any
from uuid import uuid4


@dataclass
class User:
    """
    User state and preferences.
    
    Attributes:
        id: Unique user identifier
        name: User's name (optional)
        pinned_chunk_ids: IDs of chunks user has pinned
        forgotten_chunk_ids: IDs of chunks user has forgotten
        preferences: User preferences
        created_at: Account creation time
        last_active: Last interaction
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    pinned_chunk_ids: Set[str] = field(default_factory=set)
    forgotten_chunk_ids: Set[str] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def pin_chunk(self, chunk_id: str):
        """Mark chunk as pinned"""
        self.pinned_chunk_ids.add(chunk_id)
        self.last_active = datetime.now()
    
    def forget_chunk(self, chunk_id: str):
        """Mark chunk as forgotten"""
        self.forgotten_chunk_ids.add(chunk_id)
        if chunk_id in self.pinned_chunk_ids:
            self.pinned_chunk_ids.remove(chunk_id)
        self.last_active = datetime.now()
    
    def is_pinned(self, chunk_id: str) -> bool:
        """Check if chunk is pinned"""
        return chunk_id in self.pinned_chunk_ids
    
    def is_forgotten(self, chunk_id: str) -> bool:
        """Check if chunk is forgotten"""
        return chunk_id in self.forgotten_chunk_ids
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'pinned_chunk_ids': list(self.pinned_chunk_ids),
            'forgotten_chunk_ids': list(self.forgotten_chunk_ids),
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary"""
        return cls(
            id=data['id'],
            name=data.get('name', ''),
            pinned_chunk_ids=set(data.get('pinned_chunk_ids', [])),
            forgotten_chunk_ids=set(data.get('forgotten_chunk_ids', [])),
            preferences=data.get('preferences', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_active=datetime.fromisoformat(data['last_active']),
        )