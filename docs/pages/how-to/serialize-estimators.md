# How to Serialize Estimators

This guide shows you how to save and load wrapped estimators using `joblib`. Use this when you need to persist trained models, pipelines, or `GridSearchCV` results to disk.

!!! tip "Interactive notebook available"

    Try this guide as an interactive notebook:
    [How to Serialize Estimators](/examples/serialization/)

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- A working wrapper class ([How to Wrap a Class](wrap-a-class.md))

## Save and Load a Single Estimator

Use `joblib.dump()` and `joblib.load()` exactly as with any sklearn estimator:

```python
import joblib

wrapper.fit(X_train, y_train)

# Save
joblib.dump(wrapper, "model.pkl")

# Load
loaded = joblib.load("model.pkl")
predictions = loaded.predict(X_test)
```

The loaded estimator produces identical predictions to the original.

## Save a Pipeline

Pipelines containing wrapped estimators persist all preprocessing steps:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MyWrapper(model=MyClass, alpha=1.0)),
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "pipeline.pkl")
loaded_pipeline = joblib.load("pipeline.pkl")
```

## Save GridSearchCV Results

`GridSearchCV` objects retain the best estimator, best parameters, and all cross-validation results:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(wrapper, param_grid, cv=5)
search.fit(X_train, y_train)

joblib.dump(search, "search.pkl")

loaded_search = joblib.load("search.pkl")
print(loaded_search.best_params_)
print(loaded_search.best_score_)
```

## Alternative: Use pickle

If you prefer the standard library, `pickle` works as well:

```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(wrapper, f)

with open("model.pkl", "rb") as f:
    loaded = pickle.load(f)
```

`joblib` is generally preferred for sklearn estimators because it handles large NumPy arrays more efficiently.

## See Also

- [API Reference](../reference/api.md) - `BaseClassWrapper` serialization details
- [How to Wrap a Class](wrap-a-class.md) - creating wrappers to serialize
