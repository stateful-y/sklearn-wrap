# Wrapping Functions

In this tutorial, we will wrap a standalone Python function into a scikit-learn compatible functor using `FunctionWrapper`. Along the way, we will define a wrapper subclass, call it as a functor, and then extend it into a full estimator for use with `GridSearchCV`.

!!! tip "Interactive notebook available"

    Try the concepts from this tutorial as an interactive notebook:
    [Function Wrapper](/examples/function_wrapper/)

## Prerequisites

- Python 3.11+ installed
- sklearn-wrap installed ([Getting Started](getting-started.md))

## The Keyword-Only Convention

`FunctionWrapper` identifies config parameters by inspecting your function's signature. Parameters before the `*` separator are **data** (passed at call time), parameters after `*` are **config** (stored on the wrapper and managed via `get_params`/`set_params`).

```python
def scaled_predict(X, *, scale=1.0, offset=0.0):
    """Positional args = data, keyword-only args = config."""
    return X.sum(axis=1) * scale + offset
```

Here, `X` is data (passed when calling the wrapper), while `scale` and `offset` are config (stored and tunable).

## Creating a Function Wrapper

Now we create a wrapper subclass. The only required attribute is `_callable_name`, which names the keyword argument used to pass the callable:

```python
from sklearn_wrap import FunctionWrapper

class ScaledPredictor(FunctionWrapper):
    _callable_name = "fn"
```

That's it. Let's create an instance and inspect its parameters:

```python
predictor = ScaledPredictor(fn=scaled_predict, scale=2.0, offset=1.0)
print(predictor.get_params())
```

The output should look something like:

```text
{'scale': 2.0, 'offset': 1.0, 'fn': <function scaled_predict at 0x...>}
```

Notice that `scale` and `offset` are managed as sklearn parameters, while `fn` holds the wrapped callable.

## Calling the Wrapper

Call the wrapper like a regular function. Positional data arguments are forwarded, and stored config params are injected automatically:

```python
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
result = predictor(X)
print(result)
```

The output should look something like:

```text
[ 7. 15. 23.]
```

Each prediction is `X.sum(axis=1) * 2.0 + 1.0`. Notice that `scale` and `offset` came from the wrapper's stored config, not from the call site.

## Updating Parameters

Use `set_params` to change config without creating a new wrapper:

```python
predictor.set_params(scale=0.5, offset=10.0)
print(predictor(X))
```

```text
[11.5 13.5 15.5]
```

## Cloning

The wrapper works with sklearn's `clone()`, which creates a fresh copy with the same parameters:

```python
from sklearn.base import clone

cloned = clone(predictor)
print(cloned.get_params()["scale"])  # 0.5
print(cloned.callable_fn is scaled_predict)  # True
```

## Building a Full Estimator

To use the wrapper with `Pipeline` and `GridSearchCV`, add a sklearn mixin and implement `fit`/`predict`. Use the `@_fit_context` decorator for automatic validation:

```python
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

Now let's use it with `GridSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

X_train = np.random.randn(100, 3)
y_train = X_train.sum(axis=1) * 2.0 + 1.0

grid = GridSearchCV(
    FunctionRegressor(fn=scaled_predict),
    param_grid={"scale": [0.5, 1.0, 2.0, 3.0], "offset": [0.0, 0.5, 1.0]},
    cv=3,
    scoring="neg_mean_squared_error",
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

You should see output like:

```text
Best params: {'offset': 1.0, 'scale': 2.0}
```

The exact values may vary, but notice that `GridSearchCV` found the optimal `scale` and `offset` automatically.

## What We Built

We wrapped a standalone function into an sklearn-compatible functor using `FunctionWrapper`. Along the way, we:

- Used the keyword-only convention to separate data from config parameters
- Created a wrapper subclass with a single `_callable_name` attribute
- Called the wrapper as a functor and updated its parameters
- Extended it into a full estimator with `RegressorMixin` and `_fit_context`

**Next steps:**

- [How to Wrap Functions](../how-to/wrap-functions.md): task-oriented guide covering `functools.partial`, `**kwargs`, and advanced patterns
- [About Function Wrapping](../explanation/function-wrapping.md): the functor design and how it relates to `BaseClassWrapper`
- [API Reference](../reference/api.md): full `FunctionWrapper` documentation
