# How to Wrap a Class

This guide shows you how to wrap any Python class into a Scikit-Learn compatible estimator. Use this when you have an existing class with a custom API that you want to use with `GridSearchCV`, `Pipeline`, or other Scikit-Learn tools.

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Scikit-Learn's `fit`/`predict` pattern

## Wrapping a Regressor

### 1. Define the Wrapper Class

Inherit from `BaseClassWrapper` and the appropriate Scikit-Learn mixin:

```python
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin

class MyRegressorWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "regressor"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)
```

### 2. Instantiate and Use

```python
wrapper = MyRegressorWrapper(regressor=MyRegressorClass, alpha=0.1)
wrapper.fit(X_train, y_train)
predictions = wrapper.predict(X_test)
```

The `regressor` keyword matches `_estimator_name`. All other keyword arguments become constructor parameters for the wrapped class.

## Wrapping a Classifier

If your class performs classification, use `ClassifierMixin` instead:

```python
from sklearn.base import ClassifierMixin

class MyClassifierWrapper(BaseClassWrapper, ClassifierMixin):
    _estimator_name = "classifier"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)
```

If the wrapped class exposes probability estimates, add `predict_proba`:

```python
    def predict_proba(self, X):
        return self.instance_.predict_proba(X)
```

## Wrapping a Transformer

For preprocessing or feature engineering classes, use `TransformerMixin`:

```python
from sklearn.base import TransformerMixin

class MyTransformerWrapper(BaseClassWrapper, TransformerMixin):
    _estimator_name = "transformer"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self.instance_.fit(X)
        return self

    def transform(self, X):
        return self.instance_.transform(X)
```

Note that `fit` accepts `y=None` since transformers may not need labels.

## Adapting Non-Standard Method Names

If your class uses different method names (e.g., `train` instead of `fit`), translate them in the wrapper:

```python
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.train(X, y)  # Delegates to the actual method name
        return self

    def predict(self, X):
        return self.instance_.inference(X)  # Translates predict -> inference
```

## Restricting the Wrapped Class

To enforce that only compatible classes are wrapped, set `_estimator_base_class` to a specific type:

```python
class XGBoostWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "booster"
    _estimator_base_class = xgboost.core.Booster  # Only accepts Booster subclasses
```

Passing a class that does not inherit from the specified base raises a `TypeError` at wrapper creation.

## See Also

- [About Core Concepts](../explanation/concepts.md) - understanding the delegation pattern
- [How to Validate Parameters](validate-parameters.md) - adding parameter constraints
- [How to Nest Wrappers](nest-wrappers.md) - composing wrappers hierarchically
- [API Reference](../reference/api.md) - full `BaseClassWrapper` documentation
