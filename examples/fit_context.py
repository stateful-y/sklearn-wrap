# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "sklearn-wrap",
# ]
# ///
"""
# The Fit Context Decorator

In this notebook, we compare manual instantiation with the `_fit_context` decorator
and explore how it automates validation, instantiation, and fitted state management.
"""

import marimo

__generated_with = "0.19.8"
__gallery__ = {
    "title": "The Fit Context Decorator",
    "description": "Use the _fit_context decorator to control instantiation and validation in fit methods.",
    "category": "tutorial",
    "companion": "pages/tutorials/getting-started.md",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np

    from sklearn_wrap import BaseClassWrapper
    from sklearn_wrap.base import _fit_context
    return BaseClassWrapper, np, _fit_context


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In this notebook, we compare the manual approach to instantiation and
    validation with the `_fit_context` decorator. We see how the decorator
    automates boilerplate and handles `partial_fit` for incremental learning.

    **Prerequisites:** Familiarity with the
    [first wrapper notebook](/examples/first_wrapper/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Manual vs Decorator Approach

    `_fit_context` automates sklearn's validation and fit context management.
    It handles:

    1. Parameter validation via `_validate_params()`
    2. Instantiation via `instantiate()`
    3. Context management for nested validation
    4. Setting `fitted_` attribute after successful fit

    Let's compare manual vs decorator-based approaches.
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
    ### Manual Approach (Explicit Calls)
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
    ### Decorator Approach (Automatic)

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
    return (DecoratorWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Validation Control

    The `prefer_skip_nested_validation` parameter controls validation in nested estimators.
    Most wrappers should use `True`. Meta-estimators that accept user-provided estimators
    (like GridSearchCV) should use `False`.

    For a deeper discussion, see the [Concepts page](/pages/explanation/concepts/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Let's Compare Both Approaches
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

    Notice that both produce identical results. The decorator approach removes
    the boilerplate while validation happens automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What the Decorator Gives Us

    - **Automatic instantiation:** No need to call `instantiate()` explicitly
    - **Validation:** `_validate_params()` called automatically before fit
    - **Context management:** Integrates with sklearn's `skip_parameter_validation` config
    - **Fitted flag:** Sets `fitted_` attribute automatically after successful fit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Incremental Learning with partial_fit

    The decorator handles `partial_fit` differently:
    - Only instantiates on the first call
    - Skips re-instantiation on subsequent calls
    - Does not automatically set `fitted_`
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

    Notice that `partial_fit` was called twice and the model accumulated state
    across both calls without re-instantiation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What We Built

    We compared manual instantiation with the `_fit_context` decorator and saw
    how the decorator automates boilerplate. Along the way, we:

    - Saw manual vs decorator approaches produce identical results
    - Used `prefer_skip_nested_validation` to control validation depth
    - Handled `partial_fit` for incremental learning

    **Next steps:**

    - Grid search integration:
      [View](/examples/grid_search/) · [Open in marimo](/examples/grid_search/edit/)
    - Nested wrapper parameters:
      [View](/examples/nested_wrappers/) · [Open in marimo](/examples/nested_wrappers/edit/)
    """)
    return


if __name__ == "__main__":
    app.run()
