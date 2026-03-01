"""
Utilities for hashing Hydra configs to produce deterministic output directories.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE_KEYS = ["hydra", "device", "wandb", "output_dir"]


def hash_config(
    config: Union[Dict[str, Any], DictConfig],
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> str:
    """Generate a deterministic MD5 hash from a config dict or DictConfig.

    Args:
        config: Configuration to hash.
        include_keys: If given, only these top-level keys are considered.
        exclude_keys: Additional top-level keys to drop before hashing.
            The keys ``hydra``, ``device``, ``wandb``, and ``output_dir`` are
            always excluded.

    Returns:
        Hex-digest string of the MD5 hash.
    """
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config)

    if include_keys is not None:
        config_dict = {k: config_dict[k] for k in include_keys if k in config_dict}

    keys_to_drop = list(_DEFAULT_EXCLUDE_KEYS)
    if exclude_keys:
        keys_to_drop.extend(exclude_keys)

    for key in keys_to_drop:
        config_dict.pop(key, None)

    serialised = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(serialised.encode()).hexdigest()


def get_output_dir(
    config: Union[Dict[str, Any], DictConfig],
    base_dir: str = "./outputs",
    experiment_name: Optional[str] = None,
    create_dir: bool = True,
    exclude_keys: Optional[List[str]] = None,
) -> str:
    """Return a deterministic output directory path derived from the config hash.

    The directory is named ``<experiment_name>_<config_hash>`` and is created
    under *base_dir*.  A ``config.yaml`` snapshot is written inside the
    directory on first creation.

    Args:
        config: Configuration to hash.
        base_dir: Parent directory that will contain the run folder.
        experiment_name: Name prefix for the directory.  Falls back to
            ``config.experiment.name``, then to ``"experiment"``.
        create_dir: Create the directory (and save the config snapshot) when
            it does not yet exist.
        exclude_keys: Extra top-level keys to exclude from the hash.

    Returns:
        Absolute-or-relative path to the output directory.
    """
    config_hash = hash_config(config, exclude_keys=exclude_keys)

    if experiment_name is None:
        if isinstance(config, DictConfig):
            experiment_name = OmegaConf.select(config, "experiment.name", default=None)
        if experiment_name is None:
            experiment_name = "experiment"

    output_dir = os.path.join(base_dir, f"{experiment_name}_{config_hash}")

    if create_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            if isinstance(config, DictConfig):
                f.write(OmegaConf.to_yaml(config))
            else:
                import yaml
                yaml.dump(config, f, default_flow_style=False)

    return output_dir


def find_matching_output_dir(
    config: Union[Dict[str, Any], DictConfig],
    base_dir: str = "./outputs",
    exclude_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """Find an existing output directory whose name ends with the config hash.

    Args:
        config: Configuration to match.
        base_dir: Directory to search in.
        exclude_keys: Extra top-level keys to exclude from the hash.

    Returns:
        Path to the matching directory, or ``None`` if not found.
    """
    config_hash = hash_config(config, exclude_keys=exclude_keys)

    if not os.path.isdir(base_dir):
        return None

    for entry in os.listdir(base_dir):
        if entry.endswith(config_hash):
            return os.path.join(base_dir, entry)

    return None
