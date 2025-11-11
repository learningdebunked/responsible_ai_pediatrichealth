"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration container."""
    
    # Data
    n_families: int = 100000
    age_range: list = field(default_factory=lambda: [0, 6])
    delay_prevalence: float = 0.15
    months_history: int = 24
    
    # Privacy
    epsilon: float = 0.5
    delta: float = 1e-5
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    
    # Federated Learning
    n_clients: int = 10
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Model
    model_type: str = "transformer"
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 24
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        # Flatten nested config
        flat_dict = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                flat_dict.update(values)
            else:
                flat_dict[section] = values
        
        # Update config attributes
        for key, value in flat_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
        
    Returns:
        Config object
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if default_path.exists():
            return Config.from_yaml(str(default_path))
        else:
            return Config()
    else:
        return Config.from_yaml(config_path)