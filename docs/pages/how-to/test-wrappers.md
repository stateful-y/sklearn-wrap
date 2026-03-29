# How to Test Your Wrapper

This guide shows you how to verify that a wrapper class is correctly implemented and fully compatible with Scikit-Learn's ecosystem.

## Prerequisites

- A working wrapper class ([Getting Started](../tutorials/getting-started.md))
- `pytest` installed

## Verify Basic Fit/Predict

Start with the simplest possible test: fit, predict, and check the output shape.

```python
import numpy as np
import pytest

def test_fit_predict(wrapper_class, estimator_class):
    X = np.random.randn(50, 3)
    y = np.random.randn(50)

    wrapper = wrapper_class(**{wrapper_class._estimator_name: estimator_class})
    wrapper.fit(X, y)
    predictions = wrapper.predict(X)

    assert predictions.shape == (50,)
```

## Verify Cloning

Scikit-Learn's `clone()` creates a new unfitted copy from `get_params()`. If this fails, `GridSearchCV` and other meta-estimators will not work.

```python
from sklearn.base import clone

def test_clone(fitted_wrapper):
    cloned = clone(fitted_wrapper)

    # Clone should have same params
    assert cloned.get_params() == fitted_wrapper.get_params()

    # Clone should NOT be fitted
    assert not hasattr(cloned, "instance_")
```

## Verify Parameter Round-Trip

Parameters set via `set_params()` should be reflected in `get_params()`:

```python
def test_param_roundtrip(wrapper):
    original = wrapper.get_params()
    wrapper.set_params(**original)
    assert wrapper.get_params() == original
```

## Verify Fitted State

After calling `fit()`, the wrapper should pass `check_is_fitted()`:

```python
from sklearn.utils.validation import check_is_fitted

def test_fitted_state(wrapper, X, y):
    with pytest.raises(Exception):
        check_is_fitted(wrapper)

    wrapper.fit(X, y)
    check_is_fitted(wrapper)  # Should not raise
```

## Verify GridSearchCV Compatibility

This is the most important integration test: does the wrapper work with `GridSearchCV`?

```python
from sklearn.model_selection import GridSearchCV

def test_gridsearch(wrapper, X, y, param_grid):
    search = GridSearchCV(wrapper, param_grid, cv=2, error_score="raise")
    search.fit(X, y)

    assert search.best_params_ is not None
    assert hasattr(search, "best_score_")
```

Use `error_score="raise"` during testing so failures surface immediately rather than being silently replaced with `NaN`.

## Verify Pipeline Compatibility

If your wrapper will be used inside pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def test_pipeline(wrapper, X, y):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", wrapper)])
    pipe.fit(X, y)
    predictions = pipe.predict(X)

    assert predictions.shape == (y.shape[0],)
```

## Organize Tests with Fixtures

Use pytest fixtures to avoid repeating setup across tests:

```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y

@pytest.fixture
def wrapper():
    return MyWrapper(regressor=MyEstimator, alpha=1.0)

@pytest.fixture
def fitted_wrapper(wrapper, sample_data):
    X, y = sample_data
    wrapper.fit(X, y)
    return wrapper
```

## Common Test Failures and Fixes

**`clone()` raises `TypeError`**
: The wrapper's `__init__` signature does not match `get_params()`. Ensure all constructor parameters are passed through to `BaseClassWrapper` and not stored separately.

**`check_is_fitted()` raises after `fit()`**
: The `@_fit_context` decorator was not applied to the `fit()` method, so `self.instance_` was never created.

**`GridSearchCV` fails with parameter errors**
: A parameter name contains `__` (reserved for nested access) or the parameter grid references names not in `get_params()`.

## See Also

- [How to Wrap a Class](wrap-a-class.md): creating wrapper classes
- [Troubleshooting](troubleshooting.md): common errors and solutions
- [API Reference](../reference/api.md): `BaseClassWrapper` documentation
