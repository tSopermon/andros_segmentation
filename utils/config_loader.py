import yaml
import os

def _type_cast(value: str, original_type: type):
    """Helper to cast env var string to the original type from config."""
    if original_type == bool:
        return value.lower() in ('true', '1', 't', 'yes', 'y')
    if original_type == int:
        return int(value)
    if original_type == float:
        return float(value)
    # Default to string if original is None or other
    return value

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    for key, value in config.items():
        if key in os.environ:
            # Attempt to cast to the same type as the default value in config
            original_type = type(value)
            # If original value is None, we default to string, or could try to infer
            config[key] = _type_cast(os.environ[key], original_type)
            
    return config