# How to Wrap Functions

This guide shows you how to wrap standalone Python functions into scikit-learn compatible functors. Use this when you have a function whose configuration you want to manage via `get_params`/`set_params`, tune with `GridSearchCV`, or compose in a `Pipeline`.

!!! tip "Interactive notebook available"

    Try this guide as an interactive notebook:
    [Function Wrapper](/examples/function_wrapper/)

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with scikit-learn's `fit`/`predict` pattern

## Wrapping a Basic Function

### 1. Define a Function with the Keyword-Only Convention

Separate data parameters (positional) from config parameters (keyword-only, after `*`):

```python
def predict(X, *, scale=1.0, offset=0.0):
    return X.sum(axis=1) * scale + offset
```

### 2. Create a Wrapper Subclass

```python
from sklearn_wrap import FunctionWrapper

class MyPredictor(FunctionWrapper):
    _callable_name = "fn"
```

### 3. Instantiate and Call

```python
wrapper = MyPredictor(fn=predict, scale=2.0)
result = wrapper(X)  # positional data args forwarded, config injected
```

## Providing a Default Callable

If your wrapper always uses the same function, set `_callable_default` so users don't need to pass it:

```python
class MyPredictor(FunctionWrapper):
    _callable_name = "fn"
    _callable_default = predict
```

```python
# No need to pass fn=
wrapper = MyPredictor(scale=3.0)
```

## Functions with `**kwargs`

If the wrapped function accepts `**kwargs`, the wrapper allows arbitrary extra config parameters:

```python
def flexible_predict(X, *, base=1.0, **extra):
    return X * base + sum(extra.values())

wrapper = MyPredictor(fn=flexible_predict, base=2.0, bonus=5.0, penalty=1.0)
```

All extra keyword arguments are stored and managed via `get_params`/`set_params`.

## Using `functools.partial`

`FunctionWrapper` works with `functools.partial`. The signature is resolved through the partial, preserving keyword-only params with updated defaults:

```python
import functools

base_fn = lambda X, y, *, alpha=1.0, beta=2.0: X * alpha + beta

tuned_fn = functools.partial(base_fn, beta=10.0)
wrapper = MyPredictor(fn=tuned_fn, alpha=3.0)

# beta defaults to 10.0 (from partial), alpha set to 3.0
print(wrapper.get_params())
```

## Building a Full Estimator

To use the wrapper with `Pipeline`, `GridSearchCV`, and `cross_val_score`, add a scikit-learn mixin and implement `fit`/`predict`:

```python
import numpy as np
from sklearn.base import RegressorMixin
from sklearn_wrap.base import _fit_context

class FunctionRegressor(FunctionWrapper, RegressorMixin):
    _callable_name = "fn"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.callable_fn(X, **self._params)
```

The `@_fit_context` decorator handles parameter validation before `fit` runs. After a successful `fit`, the estimator is marked as fitted.

### Using in a Pipeline

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("regressor", FunctionRegressor(fn=predict, scale=1.0)),
])
pipe.fit(X_train, y_train)

# Nested parameter access
pipe.set_params(regressor__scale=3.0)
```

### Using with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    FunctionRegressor(fn=predict),
    param_grid={"scale": [0.5, 1.0, 2.0], "offset": [0.0, 1.0]},
    cv=3,
)
grid.fit(X_train, y_train)
```

## Functions with Required Parameters

If a keyword-only parameter has no default, the wrapper marks it as required. You must provide it before calling or fitting:

```python
def transform(X, *, gamma):
    return X ** gamma

wrapper = MyPredictor(fn=transform)
# wrapper(X)  # raises ValueError: requires parameter 'gamma'

wrapper = MyPredictor(fn=transform, gamma=2.0)
wrapper(X)  # works
```

## Limitations

- **Lambdas and closures** cannot be serialized with `pickle`/`joblib`. If you need serialization, use named functions defined at module level.
- **Uninspectable callables** (where `inspect.signature` fails) are rejected at construction time. Most modern C extensions ship with `.pyi` stubs that make inspection work, but some built-in functions may not be wrappable.

## See Also

- [Wrapping Functions tutorial](../tutorials/wrapping-functions.md): step-by-step introduction
- [About Function Wrapping](../explanation/function-wrapping.md): the functor design and its relationship to `BaseClassWrapper`
- [How to Wrap a Class](wrap-a-class.md): wrapping classes with instance lifecycle management
- [API Reference](../reference/api.md): full `FunctionWrapper` documentation
