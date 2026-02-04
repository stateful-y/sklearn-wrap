"""
# Validation and Error Handling

Learn common error patterns, parameter constraints, and how BaseClassWrapper catches mistakes early.
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sklearn.base import BaseEstimator

    from sklearn_wrap import BaseClassWrapper
    return BaseClassWrapper, BaseEstimator, mo, np


@app.cell
def _(mo):
    mo.md("""
    ## Overview

    Learn common error patterns when wrapping classes and how BaseClassWrapper provides early validation to catch mistakes before runtime. This notebook covers invalid parameters, wrong base classes, missing required parameters, reserved delimiters, and parameter constraints for type safety.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Error Pattern 1: Invalid Parameters

    BaseClassWrapper validates against the wrapped class's constructor signature.
    """)
    return


@app.class_definition
class SimpleModel:
    def __init__(self, valid_param=1.0):
        self.valid_param = valid_param


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


@app.cell(hide_code=True)
def _(SimpleWrapper, mo):
    # Utility function for error demonstrations
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
        return error_msg, display

    def invalid_param_operation():
        wrapper = SimpleWrapper(estimator_class=SimpleModel, valid_param=1.0)
        wrapper.set_params(nonexistent_param=999)

    error1, display1 = create_error_demo(
        invalid_param_operation,
        "Invalid Parameter",
        "Only parameters defined in `SimpleModel.__init__` are allowed.",
    )
    return create_error_demo, display1


@app.cell
def _(display1):
    display1()
    return


@app.cell
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


@app.class_definition
class NotSklearnClass:
    def __init__(self):
        pass


@app.cell
def _(StrictWrapper, create_error_demo):
    def wrong_base_operation():
        StrictWrapper(estimator_class=NotSklearnClass)

    error2, display2 = create_error_demo(
        wrong_base_operation,
        "Base Class Validation",
        "`NotSklearnClass` doesn't inherit from `sklearn.base.BaseEstimator`.",
    )
    return (display2,)


@app.cell
def _(display2):
    display2()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Error Pattern 3: Missing Required Parameters

    Parameters without defaults must be provided during instantiation.
    """)
    return


@app.class_definition
class ModelWithRequired:
    def __init__(self, required_param, optional_param=10):
        self.required_param = required_param
        self.optional_param = optional_param


@app.cell
def _(SimpleWrapper, create_error_demo, np):
    def missing_required_operation():
        # Create wrapper without required_param
        wrapper = SimpleWrapper(estimator_class=ModelWithRequired, optional_param=20)
        # Error happens during instantiate()
        wrapper.fit(np.array([[1, 2]]), np.array([1]))

    error3, display3 = create_error_demo(
        missing_required_operation,
        "Missing Required Parameter",
        "`required_param` has no default value and must be explicitly provided.",
    )
    return (display3,)


@app.cell
def _(display3):
    display3()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Error Pattern 4: Double Underscore Reserved

    The `__` delimiter is reserved for nested parameter syntax.
    """)
    return


@app.cell
def _(SimpleWrapper, create_error_demo):
    def reserved_delimiter_operation():
        SimpleWrapper(estimator_class=SimpleModel, param__invalid=1.0)

    error4, display4 = create_error_demo(
        reserved_delimiter_operation,
        "Reserved Delimiter",
        "Parameter names cannot contain `__` - it's reserved for nested params like `model__alpha`.",
    )
    return (display4,)


@app.cell
def _(display4):
    display4()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Error Pattern 5: Parameter Constraints

    Use `_parameter_constraints` to enforce type and inheritance requirements on parameters.
    """)
    return


@app.cell
def _(np):
    class SimpleRegressor:
        def __init__(self, scale=1.0):
            self.scale = scale

        def fit(self, X, y):
            self.mean_ = y.mean()
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.mean_ * self.scale)
    return (SimpleRegressor,)


@app.cell
def _(np):
    class EnsembleModel:
        def __init__(self, inner_estimator):
            self.inner_estimator = inner_estimator

        def fit(self, X, y):
            self.inner_estimator.fit(X, y)
            return self

        def predict(self, X):
            return self.inner_estimator.predict(X)
    return (EnsembleModel,)


@app.cell
def _(BaseClassWrapper):
    class RegressorWrapper(BaseClassWrapper):
        _estimator_name = "regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.fit(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return (RegressorWrapper,)


@app.cell
def _(BaseClassWrapper):
    class EnsembleWrapper(BaseClassWrapper):
        _estimator_name = "ensemble"
        _estimator_base_class = object
        _parameter_constraints = {
            "inner_estimator": [{"wrapper_base_class": BaseClassWrapper}],
        }

        def fit(self, X, y):
            self.instantiate()
            self.instance_.fit(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return (EnsembleWrapper,)


@app.cell
def _(EnsembleWrapper, RegressorWrapper, SimpleRegressor, create_error_demo):
    def non_wrapper_constraint_operation():
        # This fails: SimpleRegressor is not a BaseClassWrapper
        raw_estimator = SimpleRegressor(scale=1.0)
        EnsembleWrapper(estimator_class=EnsembleModel, inner_estimator=raw_estimator)

    error5, display5 = create_error_demo(
        non_wrapper_constraint_operation,
        "Type Constraint Violation",
        "`inner_estimator` must be a BaseClassWrapper instance per `_parameter_constraints`.",
    )
    return (display5,)


@app.cell
def _(display5):
    display5()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Valid Constrained Usage
    """)
    return


@app.cell
def _(EnsembleWrapper, RegressorWrapper, SimpleRegressor, np):
    # This works: inner_estimator is a BaseClassWrapper
    inner_wrapper = RegressorWrapper(estimator_class=SimpleRegressor, scale=1.5)
    valid_ensemble = EnsembleWrapper(estimator_class=EnsembleModel, inner_estimator=inner_wrapper)

    # Test it
    X_ensemble = np.array([[1], [2], [3]])
    y_ensemble = np.array([10, 20, 30])
    valid_ensemble.fit(X_ensemble, y_ensemble)
    ensemble_predictions = valid_ensemble.predict(X_ensemble)
    return (ensemble_predictions,)


@app.cell
def _(mo, ensemble_predictions):
    mo.md(f"""
    ✓ Valid ensemble created and fitted successfully

    **Predictions:** {ensemble_predictions}

    Constraints ensure type safety and proper composition patterns.
    """)
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [gridsearch.py](gridsearch.py) - Use wrapped estimators with GridSearchCV for automated hyperparameter tuning
    """)
    return


if __name__ == "__main__":
    app.run()
