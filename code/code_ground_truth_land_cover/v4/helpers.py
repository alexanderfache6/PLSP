import os
from pathlib import Path


def resolve_config_path(root, *parts):
    """Expand a config path and join sub-paths onto it."""
    return Path(str(root)).expanduser().joinpath(*parts)


def expand_path(root, *parts):
    return os.path.join(os.path.expanduser(str(root)), *parts)
