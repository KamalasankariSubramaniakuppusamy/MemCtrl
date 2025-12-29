"""
MemoryController: Main API for MemCtrl
"""

from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime

from .models import Chunk, Session, User, ChunkType, ChunkPriority
from .core.tiers import TierManager
from .config import get_config


class MemoryController:
    """
    Main controller for MemCtrl memory management.
    
    This is the primary API that users interact with.
    Handles chat, pin, forget, and memory management operations.
    """
    
    def __init__(self, 
                 user_id: Optional[str] = None,
                 control_mode: str = "hybrid",
                 config_path: Optional[str] = None):
        """
        Initialize MemoryController.
        
        Args:
            user_id: User ID (creates new if None)
            control_mode: 'automatic', 'hybrid', or 'manual'
            config_path: Path to config file (optional)
        """
        # Load config
        if config_path:
            from .config import MemCtrlConfig, set_config
            config = MemCtrlConfig.from_yaml(config_path)
            set_config(config)
        
        self.config = get_config()
        self.control_mode = control_mode
        
        # Initialize tier manager
        self.tier_manager = TierManager()
        
        # Get or create user
        self.user_id = user_id or str(uuid4())
        self.user = self._load_or_create_user(self.user_id)
        
        # Current session
        self.current_session: Optional[Session] = None
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
    
    def _load_or_create_user(self, user_id: str) -> User:
        """Load user from storage or create new"""
        user = self.tier_manager.tier2.store.retrieve_user(user_id)
        if not user:
            user = User(id=user_id)
            self.tier_manager.tier2.store.store_user(user)
        return user
    
    def _get_or_create_session(self) -> Session:
        """Get current session or create new one"""
        if self.current_session and self.current_session.is_active:
            return self.current_session
        
        # Create new session
        session = Session(
            id=str(uuid4()),
            user_id=self.user_id
        )
        self.current_session = session
        self.tier_manager.tier2.store.store_session(session)
        
        return session
    
    def _create_chunk(self, content: str, chunk_type: ChunkType = ChunkType.CONVERSATION) -> Chunk:
        """Create a new chunk"""
        # Simple token counting (rough estimate)
        tokens = len(content.split())
        
        chunk = Chunk(
            id=str(uuid4()),
            content=content,
            tokens=tokens,
            chunk_type=chunk_type,
            timestamp=datetime.now()
        )
        
        return chunk
    
    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit log"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'user_id': self.user_id,
            'details': details
        })
    
    # ==================== MAIN API METHODS ====================
    
    def chat(self, query: str) -> str:
        """
        Chat with LLM using managed memory.
        
        Args:
            query: User query
            
        Returns:
            LLM response
        """
        session = self._get_or_create_session()
        
        # Create chunk for user query
        query_chunk = self._create_chunk(f"User: {query}")
        
        # Add to memory
        self.tier_manager.add_chunk(
            query_chunk,
            user_id=self.user_id,
            session_id=session.id
        )
        
        session.add_chunk(query_chunk)
        
        # TODO: Generate actual LLM response
        # For now, simple echo
        response = f"[Echo] You said: {query}"
        
        # Create chunk for assistant response
        response_chunk = self._create_chunk(f"Assistant: {response}")
        
        # Add to memory
        self.tier_manager.add_chunk(
            response_chunk,
            user_id=self.user_id,
            session_id=session.id
        )
        
        session.add_chunk(response_chunk)
        
        # Update session
        self.tier_manager.tier2.store.store_session(session)
        
        # Check if memory is getting full
        if self.tier_manager.tier0.is_full():
            self._handle_memory_pressure()
        
        # Log action
        self._log_action('chat', {
            'query': query,
            'response': response,
            'session_id': session.id
        })
        
        return response
    
    def pin(self, content: str, note: Optional[str] = None) -> Dict[str, Any]:
        """
        Pin content to permanent memory.
        
        Args:
            content: Content to pin
            note: Optional user note about why this is pinned
            
        Returns:
            Success status and chunk info
        """
        session = self._get_or_create_session()
        
        # Create pinned chunk
        chunk = self._create_chunk(content)
        chunk.is_pinned = True
        chunk.priority = ChunkPriority.USER_PINNED
        
        if note:
            chunk.metadata['user_note'] = note
        
        # Add to memory (force=True since it's pinned)
        self.tier_manager.add_chunk(
            chunk,
            user_id=self.user_id,
            session_id=session.id
        )
        
        # Update user pins
        self.user.pin_chunk(chunk.id)
        self.tier_manager.tier2.store.store_user(self.user)
        
        # Log action
        self._log_action('pin', {
            'chunk_id': chunk.id,
            'content': content,
            'note': note
        })
        
        return {
            'success': True,
            'chunk_id': chunk.id,
            'message': '✓ Pinned to permanent memory'
        }
    
    def forget(self, query: str, confirm: bool = True) -> Dict[str, Any]:
        """
        Forget chunks matching query.
        
        Args:
            query: Search query for chunks to forget
            confirm: If True, return matches for confirmation before deleting
            
        Returns:
            Matches (if confirm=True) or deletion result
        """
        # Search for matching chunks
        matches = self.tier_manager.tier2.search(query, user_id=self.user_id)
        
        if not matches:
            return {
                'success': False,
                'message': 'No matching chunks found',
                'matches': []
            }
        
        # If confirm mode, return matches for user approval
        if confirm:
            return {
                'success': True,
                'confirm_required': True,
                'matches': [
                    {
                        'chunk_id': c.id,
                        'content': c.content[:100] + '...' if len(c.content) > 100 else c.content,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in matches
                ],
                'message': f'Found {len(matches)} chunks. Call forget_confirmed() to delete.'
            }
        
        # Delete all matches
        for chunk in matches:
            self.tier_manager.remove_chunk(chunk.id)
            self.user.forget_chunk(chunk.id)
        
        # Update user
        self.tier_manager.tier2.store.store_user(self.user)
        
        # Log action
        self._log_action('forget', {
            'query': query,
            'num_deleted': len(matches)
        })
        
        return {
            'success': True,
            'num_deleted': len(matches),
            'message': f'✓ Forgot {len(matches)} chunks'
        }
    
    def forget_confirmed(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """
        Forget specific chunks after user confirmation.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Deletion result
        """
        for chunk_id in chunk_ids:
            self.tier_manager.remove_chunk(chunk_id)
            self.user.forget_chunk(chunk_id)
        
        # Update user
        self.tier_manager.tier2.store.store_user(self.user)
        
        # Log action
        self._log_action('forget_confirmed', {
            'chunk_ids': chunk_ids,
            'num_deleted': len(chunk_ids)
        })
        
        return {
            'success': True,
            'num_deleted': len(chunk_ids),
            'message': f'✓ Forgot {len(chunk_ids)} chunks'
        }
    
    def temporary(self, content: str) -> Dict[str, Any]:
        """
        Add content to temporary memory (deleted on session end).
        """
        session = self._get_or_create_session()
        
        # Create temporary chunk
        chunk = self._create_chunk(content)
        chunk.is_temporary = True
        
        # Add to memory
        self.tier_manager.add_chunk(
            chunk,
            user_id=self.user_id,
            session_id=session.id
        )
        
        # Add to session  <-- ADD THIS LINE
        session.add_chunk(chunk)
        
        # Log action
        self._log_action('temporary', {
            'chunk_id': chunk.id,
            'content': content
        })
        
        return {
            'success': True,
            'chunk_id': chunk.id,
            'message': '✓ Added to session memory (temporary)'
        }
    
    def show_memory(self, category: str = 'all') -> Dict[str, Any]:
        """
        Show current memory state.
        
        Args:
            category: 'all', 'pinned', 'session', or 'ai_managed'
            
        Returns:
            Memory contents organized by category
        """
        result = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        if category in ['all', 'pinned']:
            # Get pinned chunks
            pinned = self.tier_manager.tier2.get_pinned(self.user_id)
            result['pinned'] = [
                {
                    'chunk_id': c.id,
                    'content': c.content,
                    'timestamp': c.timestamp.isoformat(),
                    'note': c.metadata.get('user_note')
                }
                for c in pinned
            ]
        
        if category in ['all', 'session']:
            # Get current session chunks
            if self.current_session:
                session_chunks = self.current_session.get_recent_chunks(10)
                result['session'] = [
                    {
                        'chunk_id': c.id,
                        'content': c.content[:100] + '...' if len(c.content) > 100 else c.content,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in session_chunks
                ]
        
        if category in ['all', 'ai_managed']:
            # Get AI-managed chunks from Tier 1
            ai_chunks = self.tier_manager.tier1.get_all()
            result['ai_managed'] = [
                {
                    'chunk_id': c.id,
                    'importance': c.importance_score,
                    'task_type': c.task_type,
                    'summary': c.summary
                }
                for c in ai_chunks[:10]  # Limit to 10
            ]
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Statistics for all tiers
        """
        tier_stats = self.tier_manager.get_all_stats()
        
        # Add user-specific stats
        user_stats = self.tier_manager.tier2.get_stats(self.user_id)
        
        return {
            'user_id': self.user_id,
            'control_mode': self.control_mode,
            'tiers': tier_stats,
            'user': user_stats,
            'current_session_active': self.current_session is not None and self.current_session.is_active
        }
    
    def export_data(self, format: str = 'json') -> str:
        """
        Export all user data.
        
        Args:
            format: Export format ('json' or 'text')
            
        Returns:
            Exported data as string
        """
        import json
        
        data = {
            'user_id': self.user_id,
            'export_timestamp': datetime.now().isoformat(),
            'pinned_chunks': [],
            'sessions': [],
            'audit_log': self.audit_log
        }
        
        # Get all pinned chunks
        pinned = self.tier_manager.tier2.get_pinned(self.user_id)
        data['pinned_chunks'] = [c.to_dict() for c in pinned]
        
        # Get all sessions
        sessions = self.tier_manager.tier2.store.get_user_sessions(self.user_id)
        data['sessions'] = [s.to_dict() for s in sessions]
        
        if format == 'json':
            return json.dumps(data, indent=2)
        else:
            # Simple text format
            lines = [
                f"MemCtrl Data Export - User: {self.user_id}",
                f"Exported: {data['export_timestamp']}",
                "",
                f"Pinned Chunks: {len(data['pinned_chunks'])}",
                f"Sessions: {len(data['sessions'])}",
                f"Audit Log Entries: {len(data['audit_log'])}",
            ]
            return '\n'.join(lines)
    
    def close_session(self):
        """Close current session and cleanup temporary chunks"""
        if self.current_session:
            self.current_session.close()
            self.tier_manager.tier2.store.store_session(self.current_session)
            self.current_session = None
    
    # ==================== INTERNAL METHODS ====================
    
    def _handle_memory_pressure(self):
        """
        Handle memory pressure when Tier 0 is getting full.
        
        Strategy depends on control mode:
        - automatic: Auto-evict based on policy
        - hybrid: Suggest evictions to user
        - manual: Do nothing, let user manage
        """
        if self.control_mode == 'automatic':
            # Auto-evict low-priority chunks from Tier 0
            self._auto_evict()
        
        elif self.control_mode == 'hybrid':
            # TODO: Generate suggestions for user
            # For now, auto-evict
            self._auto_evict()
        
        # Manual mode: do nothing
    
    def _auto_evict(self):
        """Automatically evict low-priority chunks from Tier 0 to Tier 1"""
        tier0_chunks = self.tier_manager.tier0.get_all()
        
        # Sort by priority (lowest first)
        sorted_chunks = sorted(
            tier0_chunks,
            key=lambda c: c.get_priority_value()
        )
        
        # Evict lowest 20% that aren't pinned
        num_to_evict = max(1, len(sorted_chunks) // 5)
        
        evicted = 0
        for chunk in sorted_chunks:
            if chunk.is_pinned:
                continue
            
            # Demote to Tier 1
            if self.tier_manager.demote_to_tier1(chunk.id):
                evicted += 1
            
            if evicted >= num_to_evict:
                break
        
        # Log action
        self._log_action('auto_evict', {
            'num_evicted': evicted,
            'reason': 'memory_pressure'
        })