import yaml
import os

def load_config():
    """
    Load configuration from config.yaml file in the project root.
    """
    # Get the directory where this script is located (python/config)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the project root (Stock_Market_Prediction)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, 'config.yaml')

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        # Return a default configuration or raise error
        raise
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        raise

# Load config once when module is imported
config = load_config()
