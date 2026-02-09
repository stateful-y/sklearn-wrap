# Getting Started

This guide will help you install and start using Sklearn-Wrap in minutes.

## Installation

### Step 1: Install the package

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install sklearn_wrap
    ```

=== "uv"

    ```bash
    uv add sklearn_wrap
    ```

=== "conda"

    ```bash
    conda install -c conda-forge sklearn_wrap
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge sklearn_wrap
    ```

> **Note**: For conda/mamba, ensure the package is published to conda-forge first.

### Step 2: Verify installation

```python
import sklearn_wrap
print(sklearn_wrap.__version__)
```

## Basic Usage

Here's a minimal example wrapping a custom polynomial regression class:

### Step 1: Define your custom class

```python
import numpy as np

class PolynomialRegressor:
    """Custom polynomial regression with gradient descent."""

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        self._degree = degree
        self._learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit_model(self, X, y):
        # Transform features to polynomial
        X_poly = np.column_stack([X**i for i in range(self._degree + 1)])

        # Initialize weights
        self.weights = np.zeros(X_poly.shape[1])

        # Gradient descent
        for _ in range(self.n_iterations):
            predictions = X_poly @ self.weights
            errors = predictions - y
            gradient = X_poly.T @ errors / len(y)
            self.weights -= self._learning_rate * gradient

        return self

    def predict_from_input(self, X):
        X_poly = np.column_stack([X**i for i in range(self.degree + 1)])
        return X_poly @ self.weights
```

### Step 2: Wrap it for Scikit-Learn

```python
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin

class PolynomialWrapper(BaseClassWrapper, RegressorMixin):
    """Sklearn-compatible wrapper for PolynomialRegressor."""

    _estimator_name = "regressor"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit_model(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict_from_input(X)
```

### Step 3: Use with Scikit-Learn tools

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() + 0.5*X.ravel()**2 + np.random.randn(100)

# Create wrapped estimator
wrapper = PolynomialWrapper(
    regressor=PolynomialRegressor,
    degree=2,
    learning_rate=0.01,
    n_iterations=1000
)

# Use with GridSearchCV
param_grid = {
    'degree': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(wrapper, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser
- Experiment with different parameters
- See visual outputs in real-time
- Download standalone HTML versions

Or run locally:

=== "just"

    ```bash
    just example
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/hello.py
    ```

## Next Steps

Now that you have Sklearn-Wrap installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand core concepts and capabilities
- **Explore examples**: Check out the [Examples](examples.md) for real-world use cases
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/sklearn-wrap/discussions) or [open an issue](https://github.com/stateful-y/sklearn-wrap/issues)
