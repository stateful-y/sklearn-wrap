import abc
import functools
import inspect
from collections import defaultdict
from typing import Any

from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _is_fitted

__all__ = ["BaseClassWrapper"]


REQUIRED_PARAM_VALUE = "__REQUIRED__"
"""Sentinel value for required parameters without defaults.

This constant is used internally to mark constructor parameters that are
required by the wrapped estimator class but have no default value. When
`instantiate()` is called, any parameter still set to this value will raise
a ValueError.

Examples
--------
>>> from sklearn_wrap.base import BaseClassWrapper, REQUIRED_PARAM_VALUE
>>> class MyWrapper(BaseClassWrapper):
...     _estimator_name = "estimator"
...     _estimator_base_class = object
>>> wrapper = MyWrapper(estimator=dict, key="value")
>>> wrapper.params
{'key': 'value'}
"""


class BaseClassWrapper(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for wrapping classes into scikit-learn estimators.

    Inheriting from this class provides default implementations of:

    - setting and getting parameters used by `GridSearchCV` and friends;
    - textual and HTML representation displayed in terminals and IDEs;
    - estimator serialization;
    - parameters validation;
    - data validation;
    - metadata routing.


    Parameters
    ----------
    **params
        The keyword argument matching ``_estimator_name`` provides the class
        to wrap (optional when ``_estimator_default_class`` is set).
        Remaining keyword arguments are passed as constructor parameters to
        the wrapped class.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_wrap.base import BaseClassWrapper, _fit_context
    >>>
    >>> # Define a simple estimator class to wrap
    >>> class SimpleRegressor:
    ...     def __init__(self, multiplier=1.0, offset=0.0):
    ...         self.multiplier = multiplier
    ...         self.offset = offset
    ...
    ...     def fit(self, X, y):
    ...         return self
    ...
    ...     def predict(self, X):
    ...         return np.full(X.shape[0], self.multiplier) + self.offset
    >>>
    >>> # Wrap it with BaseClassWrapper and use _fit_context
    >>> class MyEstimator(BaseClassWrapper):
    ...     _estimator_name = "regressor"
    ...     _estimator_base_class = object
    ...
    ...     @_fit_context(prefer_skip_nested_validation=True)
    ...     def fit(self, X, y=None):
    ...         # instantiate() is called automatically by decorator
    ...         self.instance_.fit(X, y)
    ...         return self
    ...
    ...     def predict(self, X):
    ...         return self.instance_.predict(X)
    >>>
    >>> # Use it like any sklearn estimator with parameter management
    >>> estimator = MyEstimator(regressor=SimpleRegressor, multiplier=2.0, offset=1.0)
    >>> params = estimator.get_params()
    >>> params["multiplier"]
    2.0
    >>> params["offset"]
    1.0
    >>> params["regressor"]  # doctest: +ELLIPSIS
    <class '...SimpleRegressor'>
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([3., 3., 3.])
    >>> # Parameters can be updated via set_params
    >>> estimator.set_params(multiplier=3.0, offset=0.5).fit(X, y).predict(X)
    array([3.5, 3.5, 3.5])
    """

    _required_parameters: list[str] = []
    _estimator_name: str | None = None
    _estimator_base_class = None
    _estimator_default_class: type | None = None
    _parameter_constraints: dict[str, list] = {}  # For validating parameter types

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "_estimator_name", None)
        if isinstance(name, str):
            has_default = getattr(cls, "_estimator_default_class", None) is not None
            cls._required_parameters = [] if has_default else [name]

    def __init__(self, **params):
        name = self._estimator_name
        if not isinstance(name, str):
            raise ValueError("Class should define a static `_estimator_name`.")

        if name not in params:
            default_cls = self._estimator_default_class
            if default_cls is not None:
                params[name] = default_cls
            else:
                raise TypeError(f"{self.__class__.__name__}.__init__() missing required keyword argument: '{name}'")
        estimator_class = params.pop(name)

        self.estimator_class = self._validate_estimator_class(estimator_class)
        self.params = self._validate_estimator_params(params)

        # Validate parameter constraints (including nested wrappers)
        for param_name, param_value in self.params.items():
            if param_value is not REQUIRED_PARAM_VALUE and param_value is not None:
                self._validate_nested_wrapper_param(param_name, param_value)

    @property
    def estimator_name(self) -> str:
        """Get the name of the wrapped estimator type.

        Returns
        -------
        str
            The estimator name.

        """
        if not isinstance(self._estimator_name, str):
            raise ValueError("Class should define a static `_estimator_name`.")

        return self._estimator_name

    @property
    def estimator_base_class(self) -> type:
        """Get the required base class for the wrapped estimator.

        Returns
        -------
        type
            The base class.

        """
        if self._estimator_base_class is None:
            raise ValueError("Class should define a static `_estimator_base_class`.")

        return self._estimator_base_class

    def _validate_estimator_class(self, estimator_class: type) -> type:
        """
        Validate the estimator class.

        Parameters
        ----------
        estimator_class : type
            The estimator class to validate.

        Returns
        -------
        type
            The validated estimator class.

        """
        if not inspect.isclass(estimator_class):
            raise TypeError(
                f"{self._estimator_name} parameter for estimator "
                f"{self.__class__.__name__} is not a class. It is {estimator_class!r}."
            )

        if not issubclass(estimator_class, self.estimator_base_class):
            base_class = self.estimator_base_class
            base_class_name = f"{base_class.__module__}.{base_class.__qualname__}"
            raise ValueError(
                f"Invalid {self._estimator_name} class {estimator_class.__name__!r} for estimator "
                f"{self.__class__.__name__!r}. Valid estimator class should be derived from "
                f"{base_class_name}."
            )

        return estimator_class

    def _validate_estimator_params(self, params: dict, *, validate_nested: bool = True):
        """
        Validate estimator parameters.

        Check the estimator parameter names and set the omitted ones
        to their default value as per the ``estimator_class``
        constructor.

        Parameters
        ----------
        params : dict
            Dictionary of estimator parameters. Keys should be base parameter
            names (without "__" for nested params).
        validate_nested : bool, default=True
            If False, skip validation and only return the params as-is.
            Used internally when processing already-split nested parameters.

        Returns
        -------
        dict
            Validated dictionary of estimator parameters.
        """
        if not validate_nested:
            return params

        constructor_signature = inspect.signature(self.estimator_class.__init__)

        # Check if constructor accepts **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in constructor_signature.parameters.values()
        )

        valid_class_params = {
            key: val.default
            for key, val in constructor_signature.parameters.items()
            if key != "self" and val.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }

        validated_params = {}
        for param_name, param_val in params.items():
            # Check for reserved delimiter in parameter names
            if "__" in param_name:
                raise ValueError(
                    f"Parameter name {param_name!r} cannot contain '__' (double underscore). "
                    f"This delimiter is reserved for nested parameter syntax."
                )

            # If class accepts **kwargs, allow any parameter
            if param_name not in valid_class_params and not has_var_keyword:
                raise ValueError(
                    f"{param_name!r} is not a valid parameter for class {self.estimator_class.__name__!r}."
                )

            validated_params[param_name] = param_val

        # Add default values for missing parameters (but not for *args)
        for param_name, param_val in valid_class_params.items():
            if param_name not in params:
                default_val = REQUIRED_PARAM_VALUE if param_val is inspect._empty else param_val
                validated_params[param_name] = default_val

        return validated_params

    def _validate_nested_wrapper_param(self, param_name: str, param_value: Any) -> None:
        """Validate a parameter value that should be a BaseClassWrapper.

        Checks parameter constraints defined in _parameter_constraints to ensure
        wrapped estimators have the correct base class.

        Parameters
        ----------
        param_name : str
            Name of the parameter being validated.
        param_value : Any
            The parameter value to validate.

        Raises
        ------
        TypeError
            If the value is not a BaseClassWrapper when required.
        ValueError
            If the wrapped estimator_class doesn't inherit from expected base class.
        """
        if param_name not in self._parameter_constraints:
            return

        constraints = self._parameter_constraints[param_name]
        for constraint in constraints:
            # Check if constraint specifies a required wrapper base class
            if isinstance(constraint, dict) and "wrapper_base_class" in constraint:
                required_base = constraint["wrapper_base_class"]

                # Value must be a BaseClassWrapper
                if not isinstance(param_value, BaseClassWrapper):
                    raise TypeError(
                        f"Parameter {param_name!r} must be a BaseClassWrapper instance, "
                        f"got {type(param_value).__name__!r}."
                    )

                # Check the wrapped estimator_class inheritance
                if not issubclass(param_value.estimator_class, required_base):
                    raise ValueError(
                        f"Parameter {param_name!r} must wrap an estimator class derived from "
                        f"{required_base.__module__}.{required_base.__qualname__}, "
                        f"but got {param_value.estimator_class.__name__} which derives from "
                        f"{param_value.estimator_class.__bases__}."
                    )

    def _validate_params(self):
        """Validate types and values of constructor parameters.

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        self._validate_estimator_class(self.estimator_class)
        self._validate_estimator_params(self.params)

        # Validate nested wrapper parameters according to constraints
        for param_name, param_value in self.params.items():
            if param_value is not REQUIRED_PARAM_VALUE and param_value is not None:
                self._validate_nested_wrapper_param(param_name, param_value)

    def __sklearn_is_fitted__(self) -> bool:
        """Check if the estimator has been fitted.

        This method is used by sklearn's check_is_fitted() and _is_fitted() to
        determine if an estimator has been fitted.

        Checks for fitted attributes (attributes ending with '_' excluding 'instance_'),
        which is the sklearn convention for fitted attributes. Also checks for the
        `_fitted` internal flag for backward compatibility.

        Returns
        -------
        bool
            True if the estimator has fitted attributes,
            False otherwise.
        """
        # Check internal _fitted flag first (for backward compatibility)
        if getattr(self, "_fitted", False):
            return True

        # Check for fitted attributes (excluding instance_)
        fitted_attrs = [v for v in vars(self) if v.endswith("_") and not v.startswith("__") and v != "instance_"]
        return len(fitted_attrs) > 0

    def instantiate(self) -> "BaseClassWrapper":
        """Validate parameters and create an instance.

        Returns
        -------
        self

        """
        self._validate_params()

        for param_name, param_value in self.params.items():
            if param_value == REQUIRED_PARAM_VALUE:
                raise ValueError(f"Class {self.estimator_class.__name__!r} requires parameter {param_name!r}.")

        self.instance_ = self.estimator_class(**self.params)

        # Reset fitted flag when creating a new instance
        self._fitted = False

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.

        Notes
        -----
        The estimator class is always returned under the ``_estimator_name`` key
        (e.g. ``"regressor"``, ``"classifier"``). This ensures that
        ``sklearn.base.clone()`` can reconstruct the wrapper correctly, since
        ``clone()`` passes the dict returned by ``get_params(deep=False)`` as
        keyword arguments to the constructor.
        """
        out = {}
        for key, value in self.params.items():
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                # Exclude the estimator class parameter from nested params
                # to prevent roundtrip issues (estimator class can't be set via set_params)
                if isinstance(value, BaseClassWrapper):
                    estimator_name = value._estimator_name
                    deep_items = [(k, v) for k, v in deep_items if k != estimator_name]
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value

        out[self._estimator_name] = self.estimator_class

        return out

    def set_params(self, **params: object) -> "BaseClassWrapper":
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # Check if trying to change estimator class
        if self._estimator_name in params:
            raise ValueError(
                f"Cannot change estimator class via set_params. "
                f"The '{self._estimator_name}' parameter cannot be set. Redeclare the "
                f"estimator class by creating a new instance of {self.__class__.__name__}."
            )

        # Step 1: Split parameters into simple and nested BEFORE validation
        # This is the key fix - we need to know which params are nested before validating
        simple_params = {}
        nested_params = defaultdict(dict)  # grouped by prefix
        has_nested = False

        for full_key, value in params.items():
            base_key, delim, sub_key = full_key.partition("__")
            if delim:  # Contains "__", so it's a nested parameter
                nested_params[base_key][sub_key] = value
                has_nested = True
            else:
                simple_params[base_key] = value

        # Step 2: Validate only the base/simple parameter names
        if simple_params:
            self._validate_estimator_params(simple_params)

        # Step 3: Validate base keys for nested params exist
        if has_nested:
            for base_key in nested_params:
                if base_key not in self.params:
                    raise ValueError(
                        f"Invalid parameter {base_key!r} for estimator {self}. "
                        f"Valid parameters are: {list(self.params.keys())!r}."
                    )

        # Step 4: Update simple parameters and validate type constraints
        for key, value in simple_params.items():
            # Validate nested wrapper parameters (Option B + C)
            if value is not None and value is not REQUIRED_PARAM_VALUE:
                self._validate_nested_wrapper_param(key, value)
            self.params[key] = value

        # Step 5: Recursively set nested parameters
        for base_key, sub_params in nested_params.items():
            nested_obj = self.params[base_key]
            if not hasattr(nested_obj, "set_params"):
                raise AttributeError(
                    f"Cannot set nested parameters on {base_key!r}. "
                    f"Object of type {type(nested_obj).__name__!r} does not have a set_params method."
                )
            nested_obj.set_params(**sub_params)

        return self


def _fit_context(*, prefer_skip_nested_validation):
    """Decorator to run the fit methods of estimators within context managers.

    This decorator handles automatic instantiation and validation of wrapped
    estimators before the fit method executes. It integrates with scikit-learn's
    parameter validation configuration to avoid redundant validation.

    The decorator performs the following operations:

    1. Calls `instantiate()` on the estimator (unless it's `partial_fit` on an
       already fitted estimator)
    2. Validates parameters via `_validate_params()` (unless global validation
       is disabled)
    3. Sets up a context manager to optionally skip nested validation

    Parameters
    ----------
    prefer_skip_nested_validation : bool
        If True, the validation of parameters of inner estimators or functions
        called during fit will be skipped.

        This is useful to avoid validating many times the parameters passed by the
        user from the public facing API. It's also useful to avoid validating
        parameters that we pass internally to inner functions that are guaranteed to
        be valid by the test suite.

        It should be set to True for most estimators, except for those that receive
        non-validated objects as parameters, such as meta-estimators that are given
        estimator objects.

    Returns
    -------
    decorator : callable
        A decorator function that wraps fit methods.

    Examples
    --------
    >>> from sklearn_wrap.base import BaseClassWrapper, _fit_context
    >>> class MyWrapper(BaseClassWrapper):
    ...     _estimator_name = "estimator"
    ...     _estimator_base_class = object
    ...
    ...     @_fit_context(prefer_skip_nested_validation=True)
    ...     def fit(self, X, y=None):
    ...         # instantiate() is called automatically by the decorator
    ...         self.instance_.fit(X, y)
    ...         return self

    Notes
    -----
    This decorator is particularly useful when working with nested estimators
    or meta-estimators, where parameter validation needs to be controlled
    carefully to avoid performance overhead.
    """

    def decorator(fit_method):
        """Decorate a fit method with context management.

        Parameters
        ----------
        fit_method : callable
            The fit method to wrap.

        Returns
        -------
        callable
            The wrapped fit method.
        """

        @functools.wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            """Wrap the fit method with validation and instantiation logic.

            Parameters
            ----------
            estimator : BaseClassWrapper
                The estimator instance.
            *args : tuple
                Positional arguments to pass to the fit method.
            **kwargs : dict
                Keyword arguments to pass to the fit method.

            Returns
            -------
            estimator
                The fitted estimator.
            """
            global_skip_validation = get_config()["skip_parameter_validation"]

            # we don't want to validate/instantiate again for each call to partial_fit
            partial_fit_and_fitted = fit_method.__name__ == "partial_fit" and _is_fitted(estimator)

            if not partial_fit_and_fitted and hasattr(estimator, "instantiate"):
                estimator.instantiate()

                if not global_skip_validation:
                    estimator._validate_params()

            with config_context(skip_parameter_validation=(prefer_skip_nested_validation or global_skip_validation)):
                result = fit_method(estimator, *args, **kwargs)

            # Mark as fitted after successful fit (but not for partial_fit)
            # partial_fit users should set their own fitted attributes
            if fit_method.__name__ != "partial_fit":
                estimator.fitted_ = True

            return result

        return wrapper

    return decorator
