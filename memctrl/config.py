"""
Configuration management for MemCtrl
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml


@dataclass
class MemCtrlConfig:
    """
    MemCtrl configuration
    
    Memory budgets in GB
    """
    
    # Memory budgets
    tier0_budget_gb: float = 4.0      # GPU memory
    tier1_budget_gb: float = 4.0      # RAM memory
    tier2_budget_gb: float = float('inf')  # Disk (unlimited)
    
    # Model paths
    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"
    task_classifier_path: str = "models/task_classifier.pt"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Policy network paths
    policy_medical_path: str = "models/policy_medical.pt"
    policy_code_path: str = "models/policy_code.pt"
    policy_writing_path: str = "models/policy_writing.pt"
    policy_tutoring_path: str = "models/policy_tutoring.pt"
    policy_general_path: str = "models/policy_general.pt"
    
    # Storage paths
    data_dir: str = "data/user_data"
    sqlite_path: str = "data/user_data/memory.db"
    duckdb_path: str = "data/user_data/embeddings.duckdb"
    
    # Memory management
    compression_ratio: float = 4.0    # 4:1 compression
    eviction_threshold: float = 0.9   # Evict when 90% full
    importance_threshold: float = 0.5  # Keep if importance > 0.5
    
    # Control mode
    control_mode: str = "hybrid"      # automatic, hybrid, manual
    
    # Token settings
    max_tokens_per_chunk: int = 512
    max_context_tokens: int = 4096
    
    # Task detection
    task_types: list = None
    
    def __post_init__(self):
        if self.task_types is None:
            self.task_types = ["medical", "code", "writing", "tutoring", "general"]
        
        # Create data directories if they don't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'MemCtrlConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def get_tier0_tokens(self) -> int:
        """Calculate tier 0 capacity in tokens"""
        # Rough estimate: 1GB GPU = ~250M tokens (depends on model)
        # For 7B model with float16: ~14GB total, leaves ~4GB for context
        bytes_per_token = 2  # float16
        tokens = int(self.tier0_budget_gb * 1024**3 / bytes_per_token)
        return min(tokens, self.max_context_tokens)
    
    def get_tier1_tokens(self) -> int:
        """Calculate tier 1 capacity in tokens (compressed)"""
        tier0_tokens = self.get_tier0_tokens()
        return int(tier0_tokens * self.compression_ratio)


# Global config instance
_config: Optional[MemCtrlConfig] = None


def get_config() -> MemCtrlConfig:
    """Get global configuration"""
    global _config
    if _config is None:
        # Try to load from environment or use defaults
        config_path = os.getenv('MEMCTRL_CONFIG')
        if config_path and os.path.exists(config_path):
            _config = MemCtrlConfig.from_yaml(config_path)
        else:
            _config = MemCtrlConfig()
    return _config


def set_config(config: MemCtrlConfig):
    """Set global configuration"""
    global _config
    _config = config