"""Compatibility shim — do not add constants here.
This file shadows the root 'config' package when Acquire scripts are run directly.
It re-exports acquire-specific constants from acquire_config for legacy callers.
Scripts should import from 'acquire_config' (Acquire-specific) or root 'config'
(project-wide) directly. The editable install (pip install -e .) makes both
available without any sys.path manipulation.
"""

from acquire_config import *  # noqa: F401,F403
