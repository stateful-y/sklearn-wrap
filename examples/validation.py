"""
# Validation and Error Handling

Learn common error patterns, parameter constraints, and how BaseClassWrapper catches mistakes early.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from sklearn.base import BaseEstimator

    from sklearn_wrap import BaseClassWrapper
    return BaseClassWrapper, BaseEstimator, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Overview

    Learn common error patterns when wrapping classes and how BaseClassWrapper provides early validation to catch mistakes before runtime. This notebook covers invalid parameters, wrong base classes, missing required parameters, reserved delimiters, and parameter constraints for type safety.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Error Pattern 1: Invalid Parameters

    BaseClassWrapper validates against the wrapped class's constructor signature.
    """)
    return


@app.cell
def _():
    class SimpleModel:
        """Simple model without sklearn conventions."""

        def __init__(self, valid_param=1.0):
            self._internal_param = valid_param
    return (SimpleModel,)


@app.cell
def _(BaseClassWrapper):
    class SimpleWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.fitted_ = True
            return self
    return (SimpleWrapper,)


@app.function(hide_code=True)
def create_error_demo(operation, error_title, explanation):
    error_msg = None
    try:
        operation()
    except (ValueError, TypeError) as e:
        error_msg = str(e)
    def display():
        if error_msg:
            return mo.md(f"### {error_title}\n\n```\n{error_msg}\n```\n\n{explanation}")
        return mo.md("No error was raised (unexpected)")
    return display


@app.cell
def _(SimpleModel, SimpleWrapper):
    def invalid_param_operation():
        wrapper = SimpleWrapper(estimator_class=SimpleModel, valid_param=1.0)
        wrapper.set_params(nonexistent_param=999)
    return (invalid_param_operation,)


@app.cell(hide_code=True)
def _(create_error_demo, invalid_param_operation):
    _display = create_error_demo(
        invalid_param_operation,
        "Invalid Parameter",
        "Only parameters defined in `SimpleModel.__init__` are allowed.",
    )
    _display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Error Pattern 2: Wrong Base Class

    Enforce that wrapped classes inherit from required base classes.
    """)
    return


@app.cell
def _(BaseClassWrapper, BaseEstimator):
    class StrictWrapper(BaseClassWrapper):
        _estimator_name = "estimator"
        _estimator_base_class = BaseEstimator  # Requires sklearn's BaseEstimator
    return (StrictWrapper,)


@app.cell
def _():
    class NotSklearnClass:
        def __init__(self):
            pass
    return (NotSklearnClass,)


@app.cell
def _(NotSklearnClass, StrictWrapper):
    def wrong_base_operation():
        StrictWrapper(estimator_class=NotSklearnClass)
    return (wrong_base_operation,)


@app.cell(hide_code=True)
def _(create_error_demo, wrong_base_operation):
    _display = create_error_demo(
        wrong_base_operation,
        "Base Class Validation",
        "`NotSklearnClass` doesn't inherit from `sklearn.base.BaseEstimator`.",
    )
    _display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Error Pattern 3: Missing Required Parameters

    Parameters without defaults must be provided during instantiation.
    """)
    return


@app.cell
def _():
    class ModelWithRequired:
        def __init__(self, required_param, optional_param=10):
            self.required_param = required_param
            self.optional_param = optional_param
    return (ModelWithRequired,)


@app.cell
def _(ModelWithRequired, SimpleWrapper, np):
    def missing_required_operation():
        # Create wrapper without required_param
        wrapper = SimpleWrapper(estimator_class=ModelWithRequired, optional_param=20)
        # Error happens during instantiate()
        wrapper.fit(np.array([[1, 2]]), np.array([1]))
    return (missing_required_operation,)


@app.cell(hide_code=True)
def _(create_error_demo, missing_required_operation):
    _display = create_error_demo(
        missing_required_operation,
        "Missing Required Parameter",
        "`required_param` has no default value and must be explicitly provided.",
    )
    _display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Error Pattern 4: Double Underscore Reserved

    The `__` delimiter is reserved for nested parameter syntax.
    """)
    return


@app.cell
def _(SimpleModel, SimpleWrapper):
    def reserved_delimiter_operation():
        SimpleWrapper(estimator_class=SimpleModel, param__invalid=1.0)
    return (reserved_delimiter_operation,)


@app.cell(hide_code=True)
def _(create_error_demo, reserved_delimiter_operation):
    _display = create_error_demo(
        reserved_delimiter_operation,
        "Reserved Delimiter",
        "Parameter names cannot contain `__` - it's reserved for nested params like `model__alpha`.",
    )
    _display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Error Pattern 5: Parameter Constraints

    Use `_parameter_constraints` to enforce type and inheritance requirements on parameters.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    class SimpleRegressor:
        """Non-sklearn regressor for validation examples."""

        def __init__(self, scale=1.0):
            self._scale_param = scale

        def train_model(self, X, y):
            """Train the model (not 'fit')."""
            self._computed_mean = y.mean()
            return self

        def compute_output(self, X):
            """Compute output (not 'predict')."""
            return np.full(X.shape[0], self._computed_mean * self._scale_param)
    return (SimpleRegressor,)


@app.cell
def _(np):
    class EnsembleModel:
        """Ensemble model without sklearn conventions."""

        def __init__(self, inner_estimator):
            self._wrapped_estimator = inner_estimator

        def train_ensemble(self, X, y):
            """Train the ensemble (not 'fit')."""
            self._wrapped_estimator.fit(X, y)
            return self

        def generate_output(self, X):
            """Generate output (not 'predict')."""
            return self._wrapped_estimator.predict(X)
    return (EnsembleModel,)


@app.cell
def _(BaseClassWrapper):
    class RegressorWrapper(BaseClassWrapper):
        _estimator_name = "regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_model(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.compute_output(X)
    return (RegressorWrapper,)


@app.cell
def _(BaseClassWrapper):
    class EnsembleWrapper(BaseClassWrapper):
        _estimator_name = "ensemble"
        _estimator_base_class = object
        _parameter_constraints = {
            "inner_estimator": [{"wrapper_base_class": object}],  # Wrapped estimator can be any class
        }

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_ensemble(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.generate_output(X)
    return (EnsembleWrapper,)


@app.cell
def _(EnsembleModel, EnsembleWrapper, SimpleRegressor):
    def non_wrapper_constraint_operation():
        # This fails: SimpleRegressor is not a BaseClassWrapper
        raw_estimator = SimpleRegressor(scale=1.0)
        EnsembleWrapper(estimator_class=EnsembleModel, inner_estimator=raw_estimator)
    return (non_wrapper_constraint_operation,)


@app.cell(hide_code=True)
def _(create_error_demo, non_wrapper_constraint_operation):
    _display = create_error_demo(
        non_wrapper_constraint_operation,
        "Type Constraint Violation",
        "`inner_estimator` must be a BaseClassWrapper instance per `_parameter_constraints`.",
    )
    _display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Valid Constrained Usage
    """)
    return


@app.cell
def _(EnsembleModel, EnsembleWrapper, RegressorWrapper, SimpleRegressor, np):
    # This works: inner_estimator is a BaseClassWrapper
    inner_wrapper = RegressorWrapper(estimator_class=SimpleRegressor, scale=1.5)
    valid_ensemble = EnsembleWrapper(estimator_class=EnsembleModel, inner_estimator=inner_wrapper)

    # Test it
    X_ensemble = np.array([[1], [2], [3]])
    y_ensemble = np.array([10, 20, 30])
    valid_ensemble.fit(X_ensemble, y_ensemble)
    ensemble_predictions = valid_ensemble.predict(X_ensemble)
    return (ensemble_predictions,)


@app.cell(hide_code=True)
def _(mo, ensemble_predictions):
    mo.md(f"""
    ✓ Valid ensemble created and fitted successfully

    **Predictions:** {ensemble_predictions}

    Constraints ensure type safety and proper composition patterns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Best Practices

    - Define `_estimator_base_class` to enforce inheritance requirements
    - Use `_parameter_constraints` for nested wrapper validation
    - Use `get_params()` to inspect available parameters
    - Let BaseClassWrapper validate before `instantiate()` is called
    - Provide default values in wrapped class for optional parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [grid_search.py](grid_search.py) - Use wrapped estimators with GridSearchCV for automated hyperparameter tuning
    """)
    return


if __name__ == "__main__":
    app.run()
