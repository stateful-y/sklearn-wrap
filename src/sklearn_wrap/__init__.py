"""Wrapper to convert a Python class into a scikit-learn estimator"""

from importlib.metadata import version

from .base import BaseClassWrapper

__version__ = version(__name__)

__all__ = [
    "__version__",
    "BaseClassWrapper",
]

try:
    from .config import EstimatorConfig

    __all__ += ["EstimatorConfig"]
except ImportError:  # pydantic / pyyaml not installed
    pass
