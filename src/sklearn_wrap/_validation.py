"""Shared validation utilities for BaseClassWrapper and EstimatorConfig.

Provides functions to validate class constructor parameters and dotted import
paths, avoiding duplication between the two main modules.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["validate_class_params", "validate_dotted_path"]

_REQUIRED_PARAM_VALUE = "__REQUIRED__"
"""Local alias so this module does not import from base (avoids circular deps)."""


def validate_dotted_path(dotted_path: str, *, min_segments: int = 1) -> list[str]:
    """Validate that a dotted path consists of valid Python identifiers.

    Parameters
    ----------
    dotted_path : str
        The dotted path to validate, e.g. ``"sklearn.linear_model.Ridge"``.
    min_segments : int, default=1
        Minimum number of segments required.

    Returns
    -------
    list[str]
        The individual path segments.

    Raises
    ------
    ValueError
        If the path is empty, has fewer segments than *min_segments*, or
        contains non-identifier segments.

    Examples
    --------
    >>> validate_dotted_path("sklearn.linear_model.Ridge")
    ['sklearn', 'linear_model', 'Ridge']
    >>> validate_dotted_path("sklearn.linear_model.Ridge", min_segments=2)
    ['sklearn', 'linear_model', 'Ridge']

    See Also
    --------
    validate_class_params : Validate constructor parameters against a class signature.
    """
    parts = dotted_path.split(".")
    if len(parts) < min_segments or not all(p.isidentifier() for p in parts):
        raise ValueError(f"Invalid dotted path: {dotted_path!r}")
    return parts


def validate_class_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Validate parameter names against a class constructor signature.

    Inspects the ``__init__`` signature of *cls*, checks that every key in
    *params* is an accepted parameter (or ``**kwargs`` is present), and fills
    in default values for any missing parameters.

    Parameters
    ----------
    cls : type
        The class whose constructor will be inspected.
    params : dict[str, Any]
        Parameter names and values to validate.

    Returns
    -------
    dict[str, Any]
        Validated parameters, including defaults for omitted ones. Required
        params without defaults are set to the sentinel
        ``"__REQUIRED__"``.

    Raises
    ------
    ValueError
        If *params* contains a key that is not accepted by *cls.__init__*.

    Examples
    --------
    >>> class Foo:
    ...     def __init__(self, a, b=2): ...
    >>> validate_class_params(Foo, {"a": 1})
    {'a': 1, 'b': 2}
    >>> validate_class_params(Foo, {"a": 1, "b": 3})
    {'a': 1, 'b': 3}

    See Also
    --------
    validate_dotted_path : Validate a dotted import path.
    """
    constructor_signature = inspect.signature(cls.__init__)

    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in constructor_signature.parameters.values())

    valid_class_params: dict[str, Any] = {
        key: val.default
        for key, val in constructor_signature.parameters.items()
        if key != "self" and val.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }

    validated: dict[str, Any] = {}
    for param_name, param_val in params.items():
        if param_name not in valid_class_params and not has_var_keyword:
            raise ValueError(f"{param_name!r} is not a valid parameter for class {cls.__name__!r}.")
        validated[param_name] = param_val

    for param_name, default in valid_class_params.items():
        if param_name not in params:
            validated[param_name] = _REQUIRED_PARAM_VALUE if default is inspect._empty else default

    return validated
