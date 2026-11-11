"""Configuration management and instrument parameter profile loader for autex."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from autex.logging import get_logger

logger = get_logger("config")


class Config:
    """Container for pipeline configuration parameters with dictionary-like access."""

    def __init__(self, config_dict: dict[str, Any], filepath: Path | None = None) -> None:
        self._config = config_dict
        self.filepath = filepath

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Config:
        """Load and parse a YAML configuration file.

        Parameters
        ----------
        filepath : str | Path
            Path to the target YAML file.

        Returns
        -------
        Config
            Config instance populated with settings.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        """
        path = Path(filepath).expanduser().resolve()
        if not path.is_file():
            logger.error("Configuration file not found: %s", path)
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        logger.info("Loading pipeline configuration from: %s", path)
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}

        return cls(data, filepath=path)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve a nested configuration value using dot notation.

        Example: config.get("instrument.detector.dim_x")

        Parameters
        ----------
        key_path : str
            Dot-delimited key hierarchy.
        default : Any, optional
            Default value if key is not found, by default None.

        Returns
        -------
        Any
            Retrieved value or default.
        """
        keys = key_path.split(".")
        val: Any = self._config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __repr__(self) -> str:
        return f"Config(source={self.filepath}, keys={list(self._config.keys())})"
