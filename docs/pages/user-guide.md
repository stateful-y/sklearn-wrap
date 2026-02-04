# User Guide

This guide provides comprehensive documentation for Sklearn-Wrap.

## Overview

Sklearn-wrap provides a base class (`BaseClassWrapper`) that bridges arbitrary Python classes with Scikit-Learn's estimator interface. By inheriting from `BaseClassWrapper`, you transform any class into a Scikit-Learn-compatible estimator that works with GridSearchCV, Pipeline, joblib serialization, and the entire Scikit-Learn ecosystem.

The core philosophy is **composition over inheritance**: rather than forcing you to inherit from Scikit-Learn base classes in your original implementation, Sklearn-Wrap creates a thin wrapper layer that delegates to your class. This means you can integrate third-party libraries (like XGBoost's Booster API), legacy code, or custom algorithms without modification.

When to use Sklearn-Wrap: Use it when you need Scikit-Learn compatibility but don't want to rewrite existing code, or when wrapping external libraries that don't follow Scikit-Learn conventions. For new implementations where you control the entire codebase, consider inheriting directly from Scikit-Learn's BaseEstimator instead.

## Prerequisites

Before diving into Sklearn-Wrap, it's helpful to understand:

### Scikit-Learn Ecosystem

Scikit-Learn is Python's standard machine learning library, built around the concept of estimators (objects with fit/predict methods). Sklearn-wrap integrates your code into this ecosystem, enabling use of GridSearchCV for hyperparameter tuning, Pipeline for chaining transformers, and cross_val_score for validation.

Learn more: [Scikit-Learn Documentation](https://Scikit-Learn.org/stable/)

### Python Object-Oriented Programming

Sklearn-wrap uses inheritance and metaclasses to create wrapper classes. You should be comfortable with class definitions, inheritance, and the `__init__` method. Understanding how Python handles method resolution order (MRO) helps when debugging complex wrapper hierarchies.

Learn more: [Python Classes Documentation](https://docs.python.org/3/tutorial/classes.html)

## Why Sklearn-Wrap?

Machine learning practitioners often face a dilemma: they have working implementations (custom algorithms, third-party libraries, legacy code) but want to use Scikit-Learn's powerful tooling (GridSearchCV, Pipeline, cross-validation). The traditional solution—rewriting everything to inherit from Scikit-Learn base classes—is time-consuming and error-prone.

Sklearn-wrap solves this by providing a **wrapper pattern**: you create a small adapter class that delegates to your original implementation. This preserves your code's logic and performance while exposing Scikit-Learn's parameter interface. The approach is particularly effective for integrating libraries like XGBoost's low-level Booster API or wrapping research code for production pipelines.

The solution fits into Scikit-Learn's ecosystem by implementing the estimator interface (`fit`, `predict`, `get_params`, `set_params`) while handling edge cases like nested parameters, validation, and serialization. This means wrapped estimators work identically to native Scikit-Learn estimators in all contexts.

### For Data Scientists

If you're prototyping with custom algorithms or experimenting with novel approaches, Sklearn-Wrap offers:

- **Rapid experimentation**: Wrap your custom regressor in 10 lines and immediately use GridSearchCV to tune hyperparameters, without restructuring your implementation.
- **Reproducibility**: Leverage joblib serialization to save entire pipelines (preprocessing + custom model + postprocessing) for reproducible experiments.
- **Integration**: Combine your custom estimator with Scikit-Learn's StandardScaler, PCA, or other transformers in a Pipeline, enabling end-to-end workflows.
- **Validation**: Use cross_val_score and cross-validation strategies without writing custom scoring loops.

### For ML Engineers

If you're integrating external libraries or maintaining production ML systems, you'll appreciate:

- **Library integration**: Wrap XGBoost's Booster API, LightGBM's low-level interfaces, or any third-party library to work seamlessly with your Scikit-Learn-based infrastructure.
- **Legacy code**: Adapt existing models to Scikit-Learn's interface without rewriting battle-tested implementations that have undergone extensive validation.
- **Standardization**: Enforce consistent interfaces across heterogeneous models (Scikit-Learn, XGBoost, custom) using a unified wrapper pattern, simplifying deployment and monitoring.

## Core Concepts

### The Wrapper Pattern

At its heart, Sklearn-Wrap uses a **delegation pattern**: your wrapper class holds a reference to the class you want to wrap (`estimator_class`) and its constructor parameters (`params`). When `instantiate()` is called, it creates the actual instance by calling `estimator_class(**params)` and stores it in `instance_`.

This separation enables Scikit-Learn's parameter interface to work: `get_params()` returns the constructor parameters as a dictionary, and `set_params()` updates them. The actual instance isn't created until you call `fit()`, ensuring parameter changes take effect.

The pattern requires two class attributes: `_estimator_name` (the parameter name used in `get_params()`) and `_estimator_base_class` (used to validate that `estimator_class` inherits from the expected base). This validation prevents runtime errors from incompatible classes.

```python
from sklearn_wrap.base import BaseClassWrapper

class MyWrapper(BaseClassWrapper):
    _estimator_name = "model"  # Parameter name for the wrapped class
    _estimator_base_class = object  # Required base class for validation

    def fit(self, X, y):
        self.instantiate()  # Creates instance_ from estimator_class(**params)
        self.instance_.fit(X, y)
        return self
```

### The instantiate() Method

The `instantiate()` method is the bridge from wrapper to wrapped instance. It performs three critical operations:

1. **Validation**: Calls `_validate_params()` to check parameter types and values against `_parameter_constraints`
2. **Required parameter check**: Raises `ValueError` if any parameter still has the `REQUIRED_PARAM_VALUE` sentinel
3. **Instance creation**: Calls `estimator_class(**params)` and stores the result in `instance_`

You typically don't call `instantiate()` directly—use the `_fit_context` decorator instead, which handles instantiation automatically before your `fit()` method executes. This ensures the instance is created with validated parameters.

```python
from sklearn_wrap.base import BaseClassWrapper, _fit_context

class MyWrapper(BaseClassWrapper):
    _estimator_name = "model"
    _estimator_base_class = object

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        # instantiate() called automatically before this method
        self.instance_.fit(X, y)
        return self
```

### The _fit_context Decorator

The `_fit_context` decorator wraps your `fit()` method to handle three tasks automatically:

1. **Instantiation**: Calls `instantiate()` to create `instance_` from `estimator_class` and `params`
2. **Validation**: Runs parameter validation unless globally disabled (`skip_parameter_validation` config)
3. **Nested validation control**: Sets up a context manager to optionally skip validation of nested estimators

The `prefer_skip_nested_validation` parameter controls whether nested estimators should skip their own validation. Set it to `True` for most wrappers (avoids redundant validation when parameters come from user-facing API). Set it to `False` for meta-estimators that construct parameters internally and need validation.

```python
@_fit_context(prefer_skip_nested_validation=True)  # Skip nested validation
def fit(self, X, y):
    # Decorated method receives validated, instantiated estimator
    self.instance_.fit(X, y)
    return self
```

### Parameter Interface

Sklearn's parameter interface (`get_params`/`set_params`) enables GridSearchCV and Pipeline to manipulate estimator parameters. Sklearn-wrap implements this interface with support for nested parameters using the `__` delimiter.

For example, if your wrapper contains a nested estimator (another wrapper), you can set its parameters using `outer__inner__param=value`. The double-underscore syntax creates a hierarchy that `set_params()` traverses recursively, calling `set_params()` on each level.

```python
wrapper = MyWrapper(
    estimator_class=MyClass,
    nested=AnotherWrapper(estimator_class=AnotherClass, alpha=1.0)
)

# Get all parameters (including nested)
params = wrapper.get_params(deep=True)
# Returns: {'model': MyClass, 'nested': AnotherWrapper(...), 'nested__alpha': 1.0}

# Set nested parameter
wrapper.set_params(nested__alpha=0.5)
```

## Configuration

## Configuration

### Required Class Attributes

Every concrete wrapper must define two class attributes:

**`_estimator_name`**: The parameter name used in `get_params()` to identify the wrapped class. Choose a descriptive name like `"regressor"`, `"classifier"`, or `"model"`.

**`_estimator_base_class`**: The base class that `estimator_class` must inherit from. This validates that the wrapped class has the expected interface. Use `object` for minimal validation, or a specific base class (e.g., `XGBModel`) for stricter checks.

```python
from sklearn_wrap.base import BaseClassWrapper

class MyWrapper(BaseClassWrapper):
    _estimator_name = "regressor"  # Shows as 'regressor' in get_params()
    _estimator_base_class = object  # Validates estimator_class inheritance
```

### Optional: Parameter Constraints

Define `_parameter_constraints` to validate parameter types and nested wrappers:

```python
from sklearn.utils._param_validation import Interval
from sklearn_wrap.base import BaseClassWrapper

class MyWrapper(BaseClassWrapper):
    _estimator_name = "model"
    _estimator_base_class = object
    _parameter_constraints = {
        "learning_rate": [Interval(float, 0.0, 1.0, closed="neither")],
        "inner": [{"wrapper_base_class": SomeBaseClass}],  # Validates nested wrapper
    }
```

The `wrapper_base_class` constraint ensures nested wrappers contain estimators of the expected type. This catches configuration errors before `fit()` is called.

### Advanced: Custom Validation

Override `_validate_params()` for custom validation logic:

```python
def _validate_params(self):
    super()._validate_params()  # Call parent validation first

    # Custom validation
    if self.params.get("n_estimators", 0) > 1000:
        raise ValueError("n_estimators must be <= 1000")
```

## Best Practices

### 1. Use _fit_context Decorator

**Do:**
- Decorate all `fit()` methods with `@_fit_context(prefer_skip_nested_validation=True)` to handle instantiation and validation automatically
- Set `prefer_skip_nested_validation=True` for standard wrappers that receive user parameters

**Don't:**
- Call `instantiate()` manually in decorated methods (the decorator handles it)
- Set `prefer_skip_nested_validation=False` unless your wrapper constructs parameters internally (meta-estimators)

```python
# Good: decorator handles instantiation
@_fit_context(prefer_skip_nested_validation=True)
def fit(self, X, y):
    self.instance_.fit(X, y)
    return self

# Bad: manual instantiation in decorated method
@_fit_context(prefer_skip_nested_validation=True)
def fit(self, X, y):
    self.instantiate()  # Redundant!
    self.instance_.fit(X, y)
    return self
```

### 2. Follow Sklearn Naming Conventions

**Do:**
- Use trailing underscores for fitted attributes: `self.weights_`, `self.classes_`, `self.n_features_in_`
- Store `instance_` after instantiation
- Name wrappers descriptively: `XGBoostWrapper`, `PolynomialWrapper`

**Don't:**
- Use `__` in parameter names (reserved for nested parameter syntax)
- Create fitted attributes without trailing underscores

### 3. Validate Early

Define `_parameter_constraints` to catch invalid parameters before `fit()` executes. This provides clear error messages and prevents silent failures.

```python
from sklearn.utils._param_validation import Interval, StrOptions

class MyWrapper(BaseClassWrapper):
    _parameter_constraints = {
        "n_estimators": [Interval(int, 1, None, closed="left")],
        "loss": [StrOptions({"mse", "mae", "huber"})],
    }
```

### 4. Handle Nested Estimators Carefully

When wrapping estimators that contain other estimators (ensemble methods, meta-learners), set `prefer_skip_nested_validation=False` to ensure nested parameters are validated:

```python
class EnsembleWrapper(BaseClassWrapper):
    _estimator_name = "ensemble"
    _estimator_base_class = object
    _parameter_constraints = {
        "base_estimator": [{"wrapper_base_class": BaseClassWrapper}],
    }

    @_fit_context(prefer_skip_nested_validation=False)  # Validate nested
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self
```

### 5. Document Your Wrapper

Include docstrings explaining:
- What class is being wrapped and why
- Required parameters and their defaults
- Any sklearn-specific behavior (e.g., attribute transformations)

```python
class XGBoostWrapper(BaseClassWrapper):
    """Wrap XGBoost's low-level Booster API for Scikit-Learn compatibility.

    This wrapper enables GridSearchCV and Pipeline usage with XGBoost's
    training API, which offers more control than the Scikit-Learn-compatible
    XGBRegressor/XGBClassifier interfaces.

    Parameters
    ----------
    estimator_class : class
        Should be xgboost.Booster
    max_depth : int, default=6
        Maximum tree depth
    eta : float, default=0.3
        Learning rate
    """
    _estimator_name = "booster"
    _estimator_base_class = object
```

## Limitations and Considerations

Understanding the limitations helps you make informed decisions:

1. **Estimator class immutability**: Once a wrapper is instantiated, you cannot change `estimator_class` via `set_params()`. To wrap a different class, create a new wrapper instance. This prevents ambiguity about which class's parameters are being configured.

2. **Parameter validation overhead**: Sklearn-wrap validates parameters before instantiation, which adds overhead. For performance-critical applications, consider setting `Scikit-Learn.set_config(skip_parameter_validation=True)` after validating inputs once.

3. **Metadata routing**: Sklearn-wrap does not currently implement metadata routing for sample weights or other fit parameters. You must pass these directly to the wrapped instance's methods if needed.

## Next Steps

Now that you understand the core concepts and features:

- Follow the [Getting Started](getting-started.md) guide to start using Sklearn-Wrap
- Explore the [Examples](examples.md) for real-world use cases
- Check the [API Reference](api-reference.md) for detailed API documentation
- Join the community on [GitHub Discussions](https://github.com/stateful-y/sklearn-wrap/discussions)
