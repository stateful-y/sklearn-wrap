"""Pydantic-based YAML configuration for scikit-learn estimators.

This module provides ``EstimatorConfig``, a Pydantic model that can save, load,
and validate scikit-learn estimator configurations from YAML files. It supports
YAML anchors/merge keys and a custom ``!include`` tag for multi-file composition.

Requires the ``config`` extra: ``pip install sklearn-wrap[config]``."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ._validation import validate_class_params, validate_dotted_path

# Importing `base` here is a one-way edge, not a cycle: `base` imports only
# `._validation`, never `config`, and `__init__` already imports `base`
# unconditionally. `_validation` remains the shared seam for everything else --
# this import exists solely because pruning has to know that a wrapper's real
# defaults live on the class it wraps, not on its own `**params` constructor.
from .base import BaseClassWrapper

__all__ = [
    "EstimatorConfig",
    "UntrustedModuleError",
    "config_context",
    "get_config",
    "reset_config",
    "set_config",
]

DEFAULT_TRUSTED_MODULES: frozenset[str] = frozenset({"sklearn", "sklearn_wrap"})

_global_config: dict[str, Any] = {
    "trusted_modules": DEFAULT_TRUSTED_MODULES,
}
_threadlocal = threading.local()


def get_config() -> dict[str, Any]:
    """Return a copy of the current sklearn-wrap configuration.

    Thread-local overrides (set via `set_config` or `config_context`) take
    precedence over the global defaults.

    Returns
    -------
    dict[str, Any]
        A dictionary with the current configuration values.

    Examples
    --------
    >>> cfg = get_config()
    >>> "trusted_modules" in cfg
    True

    See Also
    --------
    set_config : Persistently update the configuration.
    config_context : Temporarily update the configuration.
    """
    config = _global_config.copy()
    config.update(getattr(_threadlocal, "config", {}))
    return config


def set_config(*, trusted_modules: frozenset[str] | None = None) -> None:
    """Persistently update the sklearn-wrap configuration for the current thread.

    Parameters
    ----------
    trusted_modules : frozenset[str] or None
        If not ``None``, replace the current set of trusted top-level packages
        used by `EstimatorConfig.build`.

    Examples
    --------
    >>> set_config(trusted_modules=frozenset({"sklearn", "xgboost"}))
    >>> get_config()["trusted_modules"] == frozenset({"sklearn", "xgboost"})
    True
    >>> reset_config()  # clean up

    See Also
    --------
    get_config : Retrieve the current configuration.
    config_context : Temporarily update the configuration.
    reset_config : Restore default configuration.
    """
    local_config = getattr(_threadlocal, "config", None)
    if local_config is None:
        _threadlocal.config = {}
    if trusted_modules is not None:
        _threadlocal.config["trusted_modules"] = trusted_modules


def reset_config() -> None:
    """Restore the sklearn-wrap configuration to its defaults for the current thread.

    Examples
    --------
    >>> set_config(trusted_modules=frozenset({"my_pkg"}))
    >>> reset_config()
    >>> get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES
    True

    See Also
    --------
    set_config : Persistently update the configuration.
    get_config : Retrieve the current configuration.
    """
    _threadlocal.config = {}


@contextlib.contextmanager
def config_context(*, trusted_modules: frozenset[str] | None = None) -> Generator[None, None, None]:
    """Context manager to temporarily set sklearn-wrap configuration.

    Configuration is restored to its previous state when the context exits,
    even if an exception is raised.

    Parameters
    ----------
    trusted_modules : frozenset[str] or None
        If not ``None``, temporarily replace the trusted modules set.

    Examples
    --------
    >>> with config_context(trusted_modules=frozenset({"sklearn", "xgboost"})):
    ...     get_config()["trusted_modules"] == frozenset({"sklearn", "xgboost"})
    True
    >>> get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES
    True

    See Also
    --------
    set_config : Persistently update the configuration.
    get_config : Retrieve the current configuration.
    """
    old_config = getattr(_threadlocal, "config", {}).copy()
    try:
        set_config(trusted_modules=trusted_modules)
        yield
    finally:
        _threadlocal.config = old_config


class UntrustedModuleError(ImportError):
    """Raised when a dotted path references a module not in the trusted allowlist.

    See Also
    --------
    EstimatorConfig.build : Method that raises this error for untrusted modules.
    """


def _import_class(dotted_path: str, trusted_modules: frozenset[str] = DEFAULT_TRUSTED_MODULES) -> type:
    """Import a class from a dotted path with security allowlist.

    Parameters
    ----------
    dotted_path : str
        Fully qualified dotted path, e.g. ``"sklearn.linear_model.Ridge"``.
    trusted_modules : frozenset[str]
        Set of allowed top-level package names.

    Returns
    -------
    type
        The resolved class.

    Raises
    ------
    UntrustedModuleError
        If the top-level package is not in *trusted_modules*.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class does not exist in the module.

    Examples
    --------
    >>> cls = _import_class("sklearn.linear_model.Ridge")
    >>> cls.__name__
    'Ridge'

    See Also
    --------
    _class_to_dotted_path : Reverse operation, building a dotted path from a class.
    UntrustedModuleError : Raised when the top-level package is not trusted.
    """
    try:
        parts = validate_dotted_path(dotted_path)
    except ValueError:
        raise ImportError(f"Invalid dotted path: {dotted_path!r}") from None

    top_level = parts[0]
    if top_level not in trusted_modules:
        raise UntrustedModuleError(
            f"Module {top_level!r} is not in the trusted modules list. "
            f"Trusted: {sorted(trusted_modules)}. "
            f"Pass trusted_modules=[..., {top_level!r}] to allow it."
        )

    # Try progressively shorter module paths, with the remainder as attribute lookups
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        obj: Any = module
        for attr_name in parts[i:]:
            try:
                obj = getattr(obj, attr_name)
            except AttributeError:
                break
        else:
            if isinstance(obj, type):
                return obj
            raise ImportError(f"{dotted_path!r} resolved to {type(obj).__name__}, not a class.")

    raise ImportError(f"Could not import {dotted_path!r}.")


def _class_to_dotted_path(cls: type) -> str:
    """Build a dotted import path from a class, preferring its public re-export.

    Walks the defining module's ancestors from the top-level package downward and
    returns the first one that exposes *cls* under its own name. This yields the
    path a library documents (``sklearn.linear_model.Ridge``) rather than the
    module the class happens to be defined in
    (``sklearn.linear_model._ridge.Ridge``), so a config does not pin the
    library's private module layout.

    Only already-imported modules are consulted: every ancestor of an imported
    module is itself imported, so the walk needs no imports and has no side
    effects.

    Parameters
    ----------
    cls : type
        The class to resolve.

    Returns
    -------
    str
        Dotted path such as ``"sklearn.linear_model.Ridge"``. Falls back to
        ``f"{cls.__module__}.{cls.__qualname__}"`` when no ancestor re-exports
        the class, and for nested classes, whose ``__qualname__`` differs from
        their ``__name__``.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> _class_to_dotted_path(Ridge)
    'sklearn.linear_model.Ridge'
    >>> _class_to_dotted_path(dict)
    'builtins.dict'

    See Also
    --------
    _import_class : Reverse operation, importing a class from a dotted path.
    """
    defining_path = f"{cls.__module__}.{cls.__qualname__}"

    # A nested class is not reachable as `getattr(module, name)`, so the
    # re-export probe below would be meaningless for it.
    if cls.__qualname__ != cls.__name__:
        return defining_path

    parts = cls.__module__.split(".")
    for i in range(1, len(parts) + 1):
        module_path = ".".join(parts[:i])
        module = sys.modules.get(module_path)
        if module is not None and getattr(module, cls.__name__, None) is cls:
            return f"{module_path}.{cls.__name__}"

    return defining_path


class _IncludeLoader(yaml.SafeLoader):
    """YAML SafeLoader subclass with ``!include`` tag support.

    Included file paths are resolved relative to the including file.
    Circular includes are detected and raise ``yaml.YAMLError``.

    See Also
    --------
    _load_yaml : Public function that creates and uses this loader.
    EstimatorConfig.from_yaml : High-level method that loads YAML configs.
    """

    _include_stack: list[str] = []

    def __init__(self, stream, *, _base_dir: str | None = None):
        """Initialize the loader with a base directory for include resolution.

        Parameters
        ----------
        stream : IO
            The YAML stream to load.
        _base_dir : str or None
            Base directory for resolving relative ``!include`` paths.
            Defaults to the current working directory.
        """
        super().__init__(stream)
        self._base_dir = _base_dir or os.getcwd()


def _include_constructor(loader: _IncludeLoader, node: yaml.ScalarNode) -> Any:
    """Resolve ``!include path.yaml`` relative to the current file."""
    relative_path = loader.construct_scalar(node)
    abs_path = os.path.normpath(os.path.join(loader._base_dir, relative_path))

    resolved = os.path.realpath(abs_path)
    if resolved in _IncludeLoader._include_stack:
        raise yaml.YAMLError(f"Circular include detected: {resolved}")

    _IncludeLoader._include_stack.append(resolved)
    try:
        with open(abs_path) as f:
            child_loader = _IncludeLoader(f, _base_dir=os.path.dirname(abs_path))
            try:
                return child_loader.get_single_data()
            finally:
                child_loader.dispose()
    finally:
        _IncludeLoader._include_stack.pop()


_IncludeLoader.add_constructor("!include", _include_constructor)


def _load_yaml(path: str | Path) -> Any:
    """Load a YAML file using the include-aware safe loader.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    Any
        Parsed YAML data.

    See Also
    --------
    _IncludeLoader : The custom YAML loader used by this function.
    EstimatorConfig.from_yaml : High-level method that uses this function.
    """
    path = Path(path).resolve()
    _IncludeLoader._include_stack = []
    with open(path) as f:
        loader = _IncludeLoader(f, _base_dir=str(path.parent))
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


class _ClassRef(BaseModel):
    """Marker for a parameter whose value is a class (type), not an instance.

    Used to represent ``BaseClassWrapper``'s estimator_class parameter in YAML
    via the ``{"__type__": "dotted.path"}`` convention.

    See Also
    --------
    EstimatorConfig : The config model that uses this marker during build.
    """

    type_path: str

    @field_validator("type_path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        """Validate that the dotted path consists of valid Python identifiers.

        Parameters
        ----------
        v : str
            The dotted path to validate.

        Returns
        -------
        str
            The validated path.
        """
        validate_dotted_path(v)
        return v


class EstimatorConfig(BaseModel):
    """Configuration for a scikit-learn compatible estimator.

    Validates the structure of estimator configurations and can build
    instantiated estimators, convert existing estimators to configs,
    and serialize/deserialize to YAML.

    Attributes
    ----------
    estimator_class : str
        Fully qualified dotted import path, e.g. ``"sklearn.linear_model.Ridge"``.
    params : dict[str, Any]
        Constructor parameters. Nested dicts with an ``estimator_class`` key are
        automatically converted to nested ``EstimatorConfig`` instances. Dicts
        with a ``__type__`` key become ``_ClassRef`` markers resolved during
        ``build()``.

    Examples
    --------
    >>> config = EstimatorConfig(
    ...     estimator_class="sklearn.linear_model.Ridge",
    ...     params={"alpha": 1.0},
    ... )
    >>> est = config.build()
    >>> est.get_params()["alpha"]
    1.0
    """

    estimator_class: str
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("estimator_class")
    @classmethod
    def _validate_estimator_class(cls, v: str) -> str:
        """Validate that estimator_class is a dotted path with at least two segments.

        Parameters
        ----------
        v : str
            The dotted import path to validate.

        Returns
        -------
        str
            The validated path.
        """
        try:
            validate_dotted_path(v, min_segments=2)
        except ValueError:
            raise ValueError(
                f"estimator_class must be a valid dotted import path with at least "
                f"two segments (e.g. 'sklearn.linear_model.Ridge'), got {v!r}"
            ) from None
        return v

    @model_validator(mode="before")
    @classmethod
    def _convert_nested(cls, data: Any) -> Any:
        """Convert nested dicts with ``estimator_class`` keys into EstimatorConfig.

        Parameters
        ----------
        data : Any
            The raw data to validate.

        Returns
        -------
        Any
            The data with nested dicts converted.
        """
        if isinstance(data, dict):
            params = data.get("params")
            if isinstance(params, dict):
                data["params"] = _walk_params(params)
        return data

    def build(
        self,
        *,
        trusted_modules: frozenset[str] | None = None,
        validate_params: bool = True,
    ) -> Any:
        """Resolve the configuration into an instantiated estimator.

        Parameters
        ----------
        trusted_modules : frozenset[str] or None
            Allowed top-level packages for class resolution. When ``None``
            (the default), the value from the global configuration is used
            (see `set_config`).
        validate_params : bool, default=True
            If True, validate that the resolved parameter names match the
            constructor signature of the target class before instantiation.

        Returns
        -------
        estimator
            An instantiated scikit-learn compatible estimator.

        Examples
        --------
        >>> config = EstimatorConfig(
        ...     estimator_class="sklearn.linear_model.Ridge",
        ...     params={"alpha": 2.0},
        ... )
        >>> est = config.build()
        >>> est.alpha
        2.0

        See Also
        --------
        EstimatorConfig.from_estimator : Create a config from an existing estimator.
        EstimatorConfig.from_yaml : Load a config from a YAML file.
        set_config : Set global trusted modules configuration.
        """
        if trusted_modules is None:
            trusted_modules = get_config()["trusted_modules"]
        cls = _import_class(self.estimator_class, trusted_modules)
        resolved = _resolve_params(self.params, trusted_modules=trusted_modules)
        if validate_params:
            validate_class_params(cls, resolved)
        return cls(**resolved)

    @classmethod
    def from_estimator(cls, estimator: Any, *, prune_defaults: bool = True) -> EstimatorConfig:
        """Create a configuration from an existing estimator instance.

        Parameters
        ----------
        estimator : BaseEstimator
            A scikit-learn compatible estimator (must implement ``get_params``).
        prune_defaults : bool, default=True
            If True, omit parameters left at their constructor default, at every
            level of nesting, so the config records what the author chose rather
            than the estimator's full parameter set. If False, record every
            parameter ``get_params(deep=False)`` returns.

        Returns
        -------
        EstimatorConfig
            The configuration capturing the estimator's class and parameters.

        Notes
        -----
        A pruned config is shorter and diffs cleanly against a hand-written one,
        but it no longer pins the values it omits: it adopts whatever the library
        makes the default. If scikit-learn later ships ``n_estimators=200``, a
        pruned config that relied on today's ``100`` silently builds a different
        estimator, without error. Pass ``prune_defaults=False`` where exact
        reproducibility matters more than readability.

        Class paths are recorded at their shortest public re-export, so
        ``Ridge`` is written as ``sklearn.linear_model.Ridge`` rather than the
        private module it is defined in. Configs naming the private path still
        build.

        Examples
        --------
        >>> from sklearn.linear_model import Ridge
        >>> est = Ridge(alpha=3.0)
        >>> config = EstimatorConfig.from_estimator(est)
        >>> config.estimator_class
        'sklearn.linear_model.Ridge'
        >>> config.params
        {'alpha': 3.0}

        Everything else `Ridge` accepts is left at its default, so pruning omits
        it. Turn pruning off to capture the full set:

        >>> full = EstimatorConfig.from_estimator(est, prune_defaults=False)
        >>> full.params["alpha"]
        3.0
        >>> full.params["fit_intercept"]
        True

        See Also
        --------
        EstimatorConfig.build : Instantiate an estimator from a config.
        EstimatorConfig.to_yaml : Serialize a config to YAML.
        """
        dotted = _class_to_dotted_path(type(estimator))
        raw_params = estimator.get_params(deep=False)
        if prune_defaults:
            raw_params = _prune_defaults(estimator, raw_params)
        params = _serialize_params(raw_params, prune_defaults=prune_defaults)
        return cls(estimator_class=dotted, params=params)

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        See Also
        --------
        EstimatorConfig.from_yaml : Load a config from a YAML file.
        """
        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EstimatorConfig:
        """Load a configuration from a YAML file.

        Supports YAML anchors, merge keys (``<<: *alias``), and the
        ``!include`` tag for multi-file composition.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.

        Returns
        -------
        EstimatorConfig
            The validated configuration.

        See Also
        --------
        EstimatorConfig.to_yaml : Write a config to a YAML file.
        EstimatorConfig.build : Instantiate an estimator from a config.
        """
        data = _load_yaml(path)
        return cls.model_validate(data)


def _walk_params(params: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert nested dicts to EstimatorConfig / _ClassRef."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[key] = _walk_value(value)
    return out


def _walk_value(value: Any) -> Any:
    """Convert a single param value during model validation."""
    if isinstance(value, dict):
        if "estimator_class" in value:
            return EstimatorConfig.model_validate(value)
        if "__type__" in value:
            return _ClassRef(type_path=value["__type__"])
        # Plain dict - recurse into it for nested structures
        return {k: _walk_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_value(item) for item in value]
    return value


def _resolve_params(params: dict[str, Any], *, trusted_modules: frozenset[str]) -> dict[str, Any]:
    """Resolve params dict: build nested configs, resolve class refs, handle Pipeline tuples."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[key] = _resolve_value(value, trusted_modules=trusted_modules)
    return out


def _resolve_value(value: Any, *, trusted_modules: frozenset[str]) -> Any:
    """Resolve a single parameter value to its runtime form."""
    if isinstance(value, EstimatorConfig):
        return value.build(trusted_modules=trusted_modules)
    if isinstance(value, _ClassRef):
        return _import_class(value.type_path, trusted_modules)
    if isinstance(value, list):
        resolved = [_resolve_value(item, trusted_modules=trusted_modules) for item in value]
        # Convert 2-element [str, estimator] lists into Pipeline-compatible tuples
        return [
            (item[0], item[1]) if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) else item
            for item in resolved
        ]
    if isinstance(value, dict):
        return {k: _resolve_value(v, trusted_modules=trusted_modules) for k, v in value.items()}
    return value


def _equals_default(value: Any, default: Any) -> bool:
    """Report whether *value* is unambiguously the constructor default.

    Deliberately asymmetric: a missed match only leaves a redundant line in the
    config, while a false match silently drops a parameter the caller set. So
    anything short of certainty answers ``False``.

    The type check is what stops ``True`` matching a default of ``1``, ``0``
    matching ``0.0``, and ``0`` matching ``False``. The ``bool`` check on the
    result is what stops a numpy array, whose ``==`` returns an array rather
    than a scalar, from being read as a match.

    Parameters
    ----------
    value : Any
        The value the estimator currently holds.
    default : Any
        The value the constructor signature declares.

    Returns
    -------
    bool
        True only when the two are certainly the same value.

    Examples
    --------
    >>> _equals_default(1.0, 1.0)
    True
    >>> _equals_default(True, 1)
    False

    See Also
    --------
    _constructor_defaults : Supplies the defaults compared against here.
    """
    if value is default:
        return True
    if type(value) is not type(default):
        return False
    try:
        result = value == default
    except Exception:
        return False
    return result is True


def _constructor_defaults(estimator: Any) -> dict[str, Any]:
    """Collect the constructor defaults an estimator's parameters can be compared to.

    For a `BaseClassWrapper` the defaults live on the wrapped class, not on the
    wrapper: `BaseClassWrapper.__init__` takes ``**params`` and declares no named
    defaults, while its constructor eagerly fills in every default of
    ``estimator_class``. Reading the wrapper's own signature would therefore find
    nothing to prune on the one type that needs it most.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator whose defaults are wanted.

    Returns
    -------
    dict[str, Any]
        Parameter names mapped to their declared defaults. Parameters without a
        default are absent, as are ``*args`` and ``**kwargs``. Empty when the
        signature cannot be inspected.

    See Also
    --------
    _equals_default : Compares a live value against one of these defaults.
    """
    target = estimator.estimator_class if isinstance(estimator, BaseClassWrapper) else type(estimator)

    try:
        signature = inspect.signature(target.__init__)
    except (ValueError, TypeError):
        return {}

    defaults = {
        name: param.default
        for name, param in signature.parameters.items()
        if name != "self"
        and param.default is not inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }

    # `get_params` reports the wrapped class under the wrapper's estimator name.
    # When the wrapper declares a default class, that key has a default too and
    # is prunable: `build()` restores it from `_estimator_default_class`.
    if isinstance(estimator, BaseClassWrapper):
        default_class = estimator._estimator_default_class
        if default_class is not None:
            defaults[estimator.estimator_name] = default_class

    return defaults


def _prune_defaults(estimator: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Drop parameters left at their constructor default.

    Runs on the raw values from ``get_params(deep=False)``, before serialization.
    The order matters: a nested estimator becomes a config dict once serialized,
    and a dict never matches the live default object it came from, so pruning
    after serialization would spare exactly the nested parameters this exists to
    remove.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator the parameters came from.
    params : dict[str, Any]
        Raw parameters as returned by ``get_params(deep=False)``.

    Returns
    -------
    dict[str, Any]
        The parameters the caller did not leave at their default. Keys with no
        corresponding constructor default are always kept.

    See Also
    --------
    EstimatorConfig.from_estimator : Caller that decides whether to prune.
    """
    defaults = _constructor_defaults(estimator)
    return {
        key: value for key, value in params.items() if not (key in defaults and _equals_default(value, defaults[key]))
    }


def _serialize_params(params: dict[str, Any], *, prune_defaults: bool) -> dict[str, Any]:
    """Convert runtime param values to serializable form."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[key] = _serialize_value(value, prune_defaults=prune_defaults)
    return out


def _serialize_value(value: Any, *, prune_defaults: bool) -> Any:
    """Convert a single runtime value to its serializable form.

    *prune_defaults* is carried through rather than read from a default, so a
    nested estimator is pruned on the same terms as the one that contains it.
    """
    if inspect.isclass(value):
        return {"__type__": _class_to_dotted_path(value)}
    if hasattr(value, "get_params") and not isinstance(value, type):
        return EstimatorConfig.from_estimator(value, prune_defaults=prune_defaults).model_dump()
    if isinstance(value, tuple):
        return [_serialize_value(item, prune_defaults=prune_defaults) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, prune_defaults=prune_defaults) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v, prune_defaults=prune_defaults) for k, v in value.items()}
    return value
