<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/sklearn-wrap/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/sklearn-wrap/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/sklearn-wrap/main/docs/assets/logo_light.png" alt="Sklearn-Wrap">
  </picture>
</p>


[![Python Version](https://img.shields.io/pypi/pyversions/sklearn_wrap)](https://pypi.org/project/sklearn_wrap/)
[![License](https://img.shields.io/github/license/stateful-y/sklearn-wrap)](https://github.com/stateful-y/sklearn-wrap/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/sklearn_wrap)](https://pypi.org/project/sklearn_wrap/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sklearn_wrap)](https://anaconda.org/conda-forge/sklearn_wrap)
[![codecov](https://codecov.io/gh/stateful-y/sklearn-wrap/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/sklearn-wrap)

## What is Sklearn-Wrap?

Sklearn-wrap enables you to wrap any Python class into a scikit-learn compatible estimator without rewriting your code. Whether you're integrating XGBoost's Booster API, custom gradient descent algorithms, or third-party machine learning libraries, sklearn-wrap provides the glue layer that makes them work seamlessly with sklearn's ecosystem.

With sklearn-wrap, you gain immediate access to GridSearchCV for hyperparameter tuning, Pipeline for composable workflows, and joblib for serialization, all while maintaining your original implementation. Perfect for data scientists who want sklearn compatibility without sacrificing custom logic or performance.

## What are the features of Sklearn-Wrap?

- **Minimal boilerplate**: 10-15 lines to wrap any Python class into a full sklearn estimator
- **Auto-parameter discovery**: Constructor parameters automatically exposed for GridSearchCV/RandomizedSearchCV
- **Nested parameters**: Support sklearn's double-underscore syntax for nested estimator hierarchies
- **Full ecosystem compatibility**: Works seamlessly with Pipeline, cross-validation, and joblib serialization
- **Built-in validation**: Optional parameter constraints with automatic type/value checking before fit

## How to install Sklearn-Wrap?

Install the Sklearn-Wrap package using `pip`:

```bash
pip install sklearn_wrap
```

or using `uv`:

```bash
uv pip install sklearn_wrap
```

or using `conda`:

```bash
conda install -c conda-forge sklearn_wrap
```

or using `mamba`:

```bash
mamba install -c conda-forge sklearn_wrap
```

or alternatively, add `sklearn_wrap` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with Sklearn-Wrap?

Here's a minimal example wrapping a custom polynomial regression class:

```python
import numpy as np
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV

# Your custom class (unchanged)
class PolynomialRegressor:
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        self._degree = degree
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations

    def fit_model(self, X, y):
        # ... your implementation ...
        return self

    def predict_output(self, X):
        # ... your implementation ...
        pass

# Wrapper (just 10 lines!)
class PolynomialWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "regressor"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit_model(X, y)  # Delegate to wrapped instance
        return self

    def predict(self, X):
        return self.instance_.predict_output(X)

# Use with sklearn tools
wrapper = PolynomialWrapper(
    regressor=PolynomialRegressor,
    degree=2,
    learning_rate=0.01
)

# Immediate GridSearchCV compatibility
param_grid = {'degree': [1, 2, 3], 'learning_rate': [0.001, 0.01, 0.1]}
grid_search = GridSearchCV(wrapper, param_grid, cv=5)
grid_search.fit(X, y)
```

## How do I use Sklearn-Wrap?

Full documentation is available at [https://sklearn-wrap.readthedocs.io/](https://sklearn-wrap.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://sklearn-wrap.readthedocs.io/en/latest/pages/examples/](https://sklearn-wrap.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `just example first_wrapper.py` or `uv run marimo edit examples/first_wrapper.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/sklearn-wrap/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/sklearn-wrap/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/sklearn-wrap/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [stateful-y.io contact page](https://stateful-y.io/contact).

## Where can I learn more?

- Full documentation: [https://sklearn-wrap.readthedocs.io/](https://sklearn-wrap.readthedocs.io/)
- Interactive Examples: [https://sklearn-wrap.readthedocs.io/en/latest/pages/examples/](https://sklearn-wrap.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/sklearn-wrap/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/sklearn-wrap/blob/main/LICENSE).

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
