import os
import yaml

def load_config(config_path: str = "config/grid_config.yaml") -> dict:
    """
    Loads YAML configuration for the Grid Optimization & XAI pipeline.
    """
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, config_path)
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config
