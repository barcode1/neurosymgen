import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration for reproducible experiments"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)