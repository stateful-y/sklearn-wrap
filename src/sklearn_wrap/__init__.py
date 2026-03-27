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
    from .config import (
        EstimatorConfig,
        UntrustedModuleError,
        config_context,
        get_config,
        reset_config,
        set_config,
    )

    __all__ += [
        "EstimatorConfig",
        "UntrustedModuleError",
        "config_context",
        "get_config",
        "reset_config",
        "set_config",
    ]
except ImportError:  # pydantic / pyyaml not installed  # pragma: no cover
    pass
