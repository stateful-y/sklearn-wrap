"""
# Fit Context Decorator

Understand `_fit_context` decorator for automatic instantiation and validation in fit methods.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    from sklearn_wrap import BaseClassWrapper
    from sklearn_wrap.base import _fit_context
    return BaseClassWrapper, mo, np, _fit_context


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Overview

    Understand how the `_fit_context` decorator automates parameter validation and instantiation in fit methods. This decorator is sklearn-wrap's implementation of sklearn's fit context management pattern, handling validation, instantiation, nested validation control, and fitted state management automatically. Learn when to use `prefer_skip_nested_validation=True` versus `False`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Manual vs Decorator Approach

    `_fit_context` is sklearn-wrap's implementation of sklearn's `@validate_params` and fit context management pattern.

    Automatically handles:
    1. Parameter validation via `_validate_params()`
    2. Instantiation via `instantiate()`
    3. Context management for nested validation
    4. Setting `fitted_` attribute after successful fit

    Compare manual vs decorator-based approaches.
    """)
    return


@app.cell
def _(np):
    class SimpleModel:
        """Non-sklearn model with custom methods."""

        def __init__(self, alpha=1.0):
            self._param_alpha = alpha

        def train_model(self, X, y):
            """Train the model (not 'fit')."""
            self._trained_coefficient = self._param_alpha
            return self

        def get_predictions(self, X):
            """Get predictions (not 'predict')."""
            return np.full(X.shape[0], self._trained_coefficient)
    return (SimpleModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Manual Approach (Explicit Calls)
    """)
    return


@app.cell
def _(BaseClassWrapper):
    class ManualWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        def fit(self, X, y):
            # Must manually call these
            self._validate_params()
            self.instantiate()

            self.instance_.train_model(X, y)
            # instance_ is not considered a fitted attribute by sklearn's check_is_fitted
            # so we need to define an attribute with a trailing underscore such as fitted_
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.get_predictions(X)
    return (ManualWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decorator Approach (Automatic)

    The decorator accepts `prefer_skip_nested_validation` parameter.
    """)
    return


@app.cell
def _(BaseClassWrapper, _fit_context):
    class DecoratorWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y):
            # instantiate() called automatically by decorator
            self.instance_.train_model(X, y)
            return self

        def predict(self, X):
            return self.instance_.get_predictions(X)
            return self.instance_.predict(X)
    return (DecoratorWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Understanding Validation Control

    This parameter controls validation behavior for nested estimators:

    **When `True` (recommended for most wrappers):**
    - Skips re-validating parameters passed to inner estimators
    - Avoids redundant validation in nested hierarchies
    - Improves performance when parameters are already validated

    **When `False` (for meta-estimators):**
    - Validates parameters at every level
    - Use when accepting user-provided estimator objects

    **Example:** GridSearchCV uses `False` because it receives user estimators.
    Most custom wrappers should use `True` since parameters are validated once at construction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare Both Approaches
    """)
    return


@app.cell
def _(DecoratorWrapper, ManualWrapper, SimpleModel, np):
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])

    # Manual approach
    manual = ManualWrapper(model=SimpleModel, alpha=2.0)
    manual.fit(X, y)
    manual_pred = manual.predict(X)

    # Decorator approach
    decorator = DecoratorWrapper(model=SimpleModel, alpha=2.0)
    decorator.fit(X, y)
    decorator_pred = decorator.predict(X)
    return X, decorator_pred, manual_pred, y


@app.cell(hide_code=True)
def _(decorator_pred, manual_pred, mo):
    mo.md(f"""
    **Manual Predictions:** {manual_pred}

    **Decorator Predictions:** {decorator_pred}

    ✓ Both produce identical results

    ✓ Decorator approach reduces boilerplate

    ✓ Validation happens automatically
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decorator Benefits

    **Automatic instantiation:** No need to call `instantiate()` explicitly

    **Validation:** `_validate_params()` called automatically before fit

    **Context management:** Integrates with sklearn's `skip_parameter_validation` config

    **Fitted flag:** Sets `fitted_` attribute automatically after successful fit

    **Nested validation control:** `prefer_skip_nested_validation` parameter optimizes performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Advanced Usage

    The decorator handles `partial_fit` differently:
    - Skips re-instantiation on subsequent calls
    - Only instantiates on first call
    - Doesn't automatically set `fitted_` attribute
    """)
    return


@app.cell
def _(BaseClassWrapper, np, _fit_context):
    class IncrementalModel:
        def __init__(self):
            self.sum_ = 0.0
            self.count_ = 0

        def partial_fit(self, X, y):
            self.sum_ += y.sum()
            self.count_ += len(y)
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.sum_ / self.count_)

    class IncrementalWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        @_fit_context(prefer_skip_nested_validation=True)
        def partial_fit(self, X, y):
            self.instance_.partial_fit(X, y)
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return IncrementalModel, IncrementalWrapper


@app.cell
def _(IncrementalModel, IncrementalWrapper, X, y):
    incr = IncrementalWrapper(model=IncrementalModel)

    # Multiple partial_fit calls
    incr.partial_fit(X[:2], y[:2])
    incr.partial_fit(X[2:], y[2:])

    incr_pred = incr.predict(X)
    return (incr_pred,)


@app.cell(hide_code=True)
def _(incr_pred, mo):
    mo.md(f"""
    **Incremental Predictions:** {incr_pred}

    - `partial_fit` called multiple times
    - Decorator doesn't re-instantiate on subsequent calls
    - Model accumulates state across calls
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - `_fit_context` decorator automates validation and instantiation
    - Use `prefer_skip_nested_validation=True` for most wrappers (performance)
    - Use `prefer_skip_nested_validation=False` for meta-estimators (safety)
    - Decorator handles `partial_fit` specially for incremental learning
    - Automatically sets `fitted_` attribute after successful fit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    This completes the sklearn-wrap tutorial series. You now understand wrapping patterns, parameter interfaces, validation, sklearn integration, serialization, nested structures, and decorator mechanics.

    **Explore:** Review [first_wrapper.py](first_wrapper.py) to apply what you've learned to your own custom estimators.
    """)
    return


if __name__ == "__main__":
    app.run()
