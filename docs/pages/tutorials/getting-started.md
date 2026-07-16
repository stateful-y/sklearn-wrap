# Getting Started

In this tutorial, we will wrap a custom polynomial regression class into a Scikit-Learn compatible estimator. Along the way, we will define a wrapper class, use the `@_fit_context` decorator, and run hyperparameter tuning with `GridSearchCV`.

## Prerequisites

- Python 3.11+ installed
- A terminal or command prompt

## Installation

First, install Sklearn-Wrap:

=== "pip"

    ```bash
    pip install sklearn-wrap
    ```

=== "uv"

    ```bash
    uv add sklearn-wrap
    ```

Verify the installation:

```python
import sklearn_wrap
print(sklearn_wrap.__version__)
```

The output should look something like:

```text
0.1.0a5
```

<!-- COMPANION_NOTEBOOKS -->

## A Custom Class to Wrap

Now let's define a simple polynomial regression class. This class does not follow Scikit-Learn conventions: it uses custom method names and a non-standard constructor.

```python
import numpy as np

class PolynomialRegressor:
    """Custom polynomial regression with gradient descent."""

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        self._degree = degree
        self._learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit_model(self, X, y):
        X_poly = np.column_stack([X**i for i in range(self._degree + 1)])
        self.weights = np.zeros(X_poly.shape[1])

        for _ in range(self.n_iterations):
            predictions = X_poly @ self.weights
            errors = predictions - y
            gradient = X_poly.T @ errors / len(y)
            self.weights -= self._learning_rate * gradient

        return self

    def predict_from_input(self, X):
        X_poly = np.column_stack([X**i for i in range(self._degree + 1)])
        return X_poly @ self.weights
```

Notice that this class uses `fit_model` and `predict_from_input` rather than the standard `fit` and `predict` that Scikit-Learn expects.

## Creating the Wrapper

Now we create a wrapper class that bridges `PolynomialRegressor` into the Scikit-Learn ecosystem. We inherit from `BaseClassWrapper` and `RegressorMixin`:

```python
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin

class PolynomialWrapper(BaseClassWrapper, RegressorMixin):
    """Sklearn-compatible wrapper for PolynomialRegressor."""

    _estimator_name = "regressor"
    _estimator_base_class = PolynomialRegressor
    _estimator_default_class = PolynomialRegressor

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit_model(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict_from_input(X)
```

Let's check what we wrote:

- `_estimator_name = "regressor"` tells Sklearn-Wrap that the wrapped class is passed via the `regressor` keyword argument
- `_estimator_base_class = PolynomialRegressor` ensures only `PolynomialRegressor` (or subclasses) can be wrapped
- `_estimator_default_class = PolynomialRegressor` means we don't have to pass the class every time we construct the wrapper
- The `@_fit_context` decorator automatically creates `self.instance_` before `fit` runs
- We delegate `fit` and `predict` to the wrapped instance's methods

!!! tip "Generic wrappers"

    If you want a wrapper that accepts *any* class (not just `PolynomialRegressor`), set `_estimator_base_class = object` and omit `_estimator_default_class`. You will then need to pass the class explicitly: `PolynomialWrapper(regressor=PolynomialRegressor, ...)`.

## Fitting the Wrapper

Now let's use the wrapper with some data:

```python
import numpy as np

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3 * X.ravel() + 0.5 * X.ravel() ** 2 + np.random.randn(100)

# Create the wrapped estimator
wrapper = PolynomialWrapper(
    degree=2,
    learning_rate=0.01,
    n_iterations=1000,
)

wrapper.fit(X, y)
predictions = wrapper.predict(X)
print(f"First 5 predictions: {predictions[:5].round(2)}")
```

You should see output like:

```text
First 5 predictions: [1.98 2.34 2.71 3.09 3.48]
```

Notice that we only passed the constructor parameters (`degree`, `learning_rate`, `n_iterations`) as keyword arguments. The wrapper already knows which class to instantiate thanks to `_estimator_default_class`.

## Hyperparameter Tuning with GridSearchCV

The wrapper automatically exposes parameters to Scikit-Learn's tools. Let's use `GridSearchCV` to find the best polynomial degree and learning rate:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "degree": [1, 2, 3],
    "learning_rate": [0.001, 0.01, 0.1],
}

grid_search = GridSearchCV(
    wrapper, param_grid, cv=5, scoring="neg_mean_squared_error"
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

You should see output like:

```text
Best parameters: {'degree': 2, 'learning_rate': 0.01}
Best score: -1.023
```

The exact values will vary due to random noise, but notice that `GridSearchCV` found the degree and learning rate automatically, without any modifications to `PolynomialRegressor`.

The wrapper is ready to use with any Scikit-Learn tool that accepts an estimator.

## Next Steps

- [How to Wrap a Class](../how-to/wrap-a-class.md): detailed reference for regressor, classifier, and transformer wrappers
- [About the Delegation Pattern](../explanation/delegation-pattern.md): understand the delegation pattern and architecture
- [API Reference](../reference/api.md): full `BaseClassWrapper` documentation
- [Examples](../examples/index.md): interactive notebooks demonstrating all features
