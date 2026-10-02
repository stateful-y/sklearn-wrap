# How to Validate Parameters

This guide shows you how to add parameter validation to your wrappers using Scikit-Learn's constraint system. Use this when you want to catch invalid parameter values before `fit()` runs.

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- A working wrapper class ([How to Wrap a Class](wrap-a-class.md))

<!-- COMPANION_NOTEBOOKS -->

## Defining Parameter Constraints

### 1. Add `_parameter_constraints` to Your Wrapper

Define a dictionary mapping parameter names to lists of allowed types or value ranges:

```python
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin

class MyWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "model"
    _estimator_base_class = object
    _parameter_constraints = {
        "n_estimators": [Interval(int, 1, None, closed="left")],
        "learning_rate": [Interval(float, 0.0, 1.0, closed="neither")],
        "loss": [StrOptions({"mse", "mae", "huber"})],
    }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)
```

### 2. Validation Runs Automatically

When `@_fit_context` is used, parameter validation happens before instantiation. Invalid values raise clear error messages:

```python
wrapper = MyWrapper(model=SomeClass, n_estimators=-1)
wrapper.fit(X, y)
# ValueError: The 'n_estimators' parameter must be an int in the range [1, inf). Got -1 instead.
```

## Common Constraint Types

| Constraint | Usage | Example |
|---|---|---|
| `Interval` | Numeric ranges | `Interval(float, 0.0, 1.0, closed="both")` |
| `StrOptions` | String choices | `StrOptions({"mse", "mae"})` |
| `type` | Exact type check | `int`, `str`, `np.ndarray` |
| `None` | Allow `None` | Include `None` in the list |
| `callable` | Any callable | `callable` |

Combine multiple constraints with a list. The value must match at least one:

```python
_parameter_constraints = {
    "alpha": [Interval(float, 0.0, None, closed="left"), None],  # float >= 0 or None
}
```

## Validating Nested Wrappers

If a parameter should be another wrapper wrapping a specific class type, use the `wrapper_base_class` constraint:

```python
_parameter_constraints = {
    "base_estimator": [{"wrapper_base_class": SomeBaseClass}],
}
```

This verifies that `base_estimator` is a `BaseClassWrapper` whose `_estimator_base_class` matches `SomeBaseClass`. Configuration errors are caught at validation time rather than deep inside `fit()`.

## Custom Validation Logic

For validation beyond what constraints express, override `_validate_params()`:

```python
def _validate_params(self):
    super()._validate_params()  # Run constraint checks first

    if self.params.get("max_depth", 0) > 100 and self.params.get("n_estimators", 0) > 500:
        raise ValueError(
            "max_depth > 100 with n_estimators > 500 would be prohibitively slow"
        )
```

!!! warning
    `_parameter_constraints` uses Scikit-Learn's internal `sklearn.utils._param_validation` API, which is not part of the public API and may change between versions.

## See Also

- [Configuration Reference](../reference/configuration.md): full list of constraint types and `_parameter_constraints` format
- [About the Fit Context Lifecycle](../explanation/fit-context-lifecycle.md): understanding the validation lifecycle
- [API Reference](../reference/api.md): `BaseClassWrapper._validate_params` documentation
