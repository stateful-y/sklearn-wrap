"""Wrapper to convert a Python class into a scikit-learn estimator."""

from importlib.metadata import version

from .base import REQUIRED_PARAM_VALUE, BaseClassWrapper

__version__ = version(__name__)

__all__ = [
    "__version__",
    "BaseClassWrapper",
    "REQUIRED_PARAM_VALUE",
]

try:
    from .config import EstimatorConfig, UntrustedModuleError

    __all__ += ["EstimatorConfig", "UntrustedModuleError"]
except ImportError:  # pydantic / pyyaml not installed  # pragma: no cover
    pass
