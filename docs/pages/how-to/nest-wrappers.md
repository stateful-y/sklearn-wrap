# How to Nest Wrappers

This guide shows you how to compose wrappers that contain other wrappers, and how to control parameters at any nesting depth using the double-underscore (`__`) syntax.

## Prerequisites

- sklearn-wrap installed ([Getting Started](../tutorials/getting-started.md))
- A working wrapper class ([How to Wrap a Class](wrap-a-class.md))

## Create a Nested Structure

Pass inner wrappers as constructor parameters to an outer wrapper:

```python
from sklearn_wrap.base import BaseClassWrapper, _fit_context
from sklearn.base import RegressorMixin

class InnerWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "regressor"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)


class OuterWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "ensemble"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=False)  # Validate nested
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)


inner = InnerWrapper(regressor=SomeRegressor, scale=0.8)
outer = OuterWrapper(ensemble=SomeEnsemble, estimator1=inner, blend=0.5)
```

## Inspect Nested Parameters

Use `get_params(deep=True)` to see the full parameter hierarchy:

```python
params = outer.get_params(deep=True)
# {'ensemble': SomeEnsemble,
#  'estimator1': InnerWrapper(...),
#  'estimator1__regressor': SomeRegressor,
#  'estimator1__scale': 0.8,
#  'blend': 0.5}
```

## Set Nested Parameters

Use the `__` syntax to modify parameters at any depth:

```python
outer.set_params(estimator1__scale=1.5)
```

This calls `set_params(scale=1.5)` on the `estimator1` inner wrapper.

## Use prefer_skip_nested_validation Correctly

For outer wrappers that contain nested wrappers, set `prefer_skip_nested_validation=False` so that nested wrappers also validate their parameters:

```python
class OuterWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "ensemble"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self
```

For leaf wrappers (no nested wrappers), use `prefer_skip_nested_validation=True` to avoid redundant validation.

## Validate Nested Wrapper Types

Use the `wrapper_base_class` constraint to ensure a parameter is a wrapper wrapping a specific class:

```python
class OuterWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "ensemble"
    _estimator_base_class = object
    _parameter_constraints = {
        "base_estimator": [{"wrapper_base_class": SomeBaseClass}],
    }
```

This verifies at validation time that `base_estimator` is a `BaseClassWrapper` whose `_estimator_base_class` matches `SomeBaseClass`.

## See Also

- [Nested Wrappers example](/examples/nested_wrappers/) - interactive walkthrough
- [How to Use with GridSearchCV](use-with-gridsearch.md) - tuning nested parameters with grid search
- [How to Validate Parameters](validate-parameters.md) - the `wrapper_base_class` constraint
- [About Core Concepts](../explanation/concepts.md) - nested parameter delegation explained
