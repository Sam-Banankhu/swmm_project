import os
import yaml
from pathlib import Path

class Config:
    """Configuration management for SWMM ML pipeline."""

    def __init__(self, config_file=None):
        self.config_file = config_file or self._get_default_config()
        self._config = self._load_config()

    def _get_default_config(self):
        """Get default config file based on environment."""
        env = os.getenv('SWMM_ENV', 'development')
        config_dir = Path(__file__).parent / 'config'
        config_file = config_dir / f'{env}.yaml'

        if not config_file.exists():
            config_file = config_dir / 'default.yaml'

        return config_file

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """Set configuration value."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, file_path=None):
        """Save current configuration."""
        save_path = file_path or self.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

# Global config instance
config = Config()