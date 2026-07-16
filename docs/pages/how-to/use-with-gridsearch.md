# How to Use with GridSearchCV

This guide shows you how to tune hyperparameters of wrapped estimators using Scikit-Learn's `GridSearchCV`, `Pipeline`, and `cross_val_score`.

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- A working wrapper class ([How to Wrap a Class](wrap-a-class.md))

## Define a Parameter Grid

Wrapped estimators expose parameters through the standard `get_params()` / `set_params()` interface. Build your parameter grid using those parameter names:

```python
from sklearn.model_selection import GridSearchCV

wrapper = MyWrapper(model=MyClass, alpha=1.0, n_iters=100)

param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
    "n_iters": [50, 100, 200],
}

search = GridSearchCV(wrapper, param_grid, cv=5, scoring="neg_mean_squared_error")
search.fit(X, y)

print(search.best_params_)
print(search.best_score_)
```

## Tune Nested Parameters

For wrappers that contain nested wrappers or estimators, use the double-underscore (`__`) syntax to reach into nested levels:

```python
param_grid = {
    "estimator1__scale": [0.5, 1.0, 1.5],
    "estimator2__scale": [0.8, 1.2],
    "blend": [0.3, 0.5, 0.7],
}

search = GridSearchCV(ensemble_wrapper, param_grid, cv=3)
search.fit(X, y)
```

The `__` notation traverses the parameter hierarchy. `estimator1__scale` calls `set_params(estimator1__scale=value)` on the outer wrapper, which delegates to the inner wrapper's `set_params(scale=value)`.

## Use in a Pipeline

Wrapped estimators work in `Pipeline` like any other estimator. Prefix parameter names with the pipeline step name:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MyWrapper(model=MyClass, alpha=1.0)),
])

param_grid = {
    "model__alpha": [0.01, 0.1, 1.0],
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X, y)
```

## Use cross_val_score

For quick evaluation without parameter search:

```python
from sklearn.model_selection import cross_val_score

wrapper = MyWrapper(model=MyClass, alpha=1.0)
scores = cross_val_score(wrapper, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"Mean score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## See Also

- [Grid Search example](/examples/grid_search/): interactive walkthrough with full code
- [How to Nest Wrappers](nest-wrappers.md): composing wrappers with `__` parameter syntax
- [About the Delegation Pattern](../explanation/delegation-pattern.md): how the parameter interface enables GridSearchCV
