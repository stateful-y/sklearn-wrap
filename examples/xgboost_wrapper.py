"""
# XGBoost Integration

Wrap XGBoost's low-level Booster API for sklearn compatibility.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import xgboost as xgb

    from sklearn_wrap import BaseClassWrapper

    return BaseClassWrapper, mo, np, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Overview

    Wrap XGBoost's low-level Booster API to make it sklearn-compatible. XGBoost's native API uses DMatrix objects and doesn't follow sklearn conventions. This notebook shows how to bridge the gap by creating an adapter class that wraps `xgb.train()`, enabling XGBoost to work with GridSearchCV, Pipelines, and other sklearn tools.

    This example also demonstrates **nested wrappers** by wrapping XGBoost callbacks. The nested parameter syntax (`callbacks__period=5`) works automatically through BaseClassWrapper's built-in `get_params`/`set_params` - no manual override needed!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. XGBoost Wrapper Setup

    Create an adapter class to wrap `xgb.train()` as a regular class, then wrap it with nested callback support.
    """)
    return


@app.cell
def _(BaseClassWrapper, xgb):
    from sklearn_wrap.base import _fit_context

    class XGBoostCallbackWrapper(BaseClassWrapper):
        """Wrap XGBoost callback classes.

        Enables nested parameter control over callbacks using the __ syntax.
        """

        _estimator_name = "callback"
        _estimator_base_class = xgb.callback.TrainingCallback  # XGBoost callbacks inherit from TrainingCallback

    # Create an adapter class that wraps xgb.train() as a normal class
    class XGBoostTrainer:
        """Adapter class that wraps XGBoost's train() function as a trainable class.

        This demonstrates how to wrap procedural APIs by creating an adapter class.
        """

        def __init__(self, num_boost_round=100, callbacks=None, **xgb_params):
            self.num_boost_round = num_boost_round
            self.callbacks = callbacks
            self.xgb_params = xgb_params

        def fit_model(self, X, y):
            dtrain = xgb.DMatrix(X, label=y)

            # Instantiate wrapped callbacks if provided
            callback_instances = None
            if self.callbacks is not None:
                if isinstance(self.callbacks, list):
                    callback_instances = [
                        cb.instance_ if hasattr(cb, 'instance_') else cb.instantiate().instance_
                        for cb in self.callbacks
                    ]
                else:
                    cb = self.callbacks
                    callback_instances = [cb.instance_ if hasattr(cb, 'instance_') else cb.instantiate().instance_]

            self.booster_ = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.num_boost_round,
                callbacks=callback_instances
            )
            return self

        def predict_output(self, X):
            dtest = xgb.DMatrix(X)
            return self.booster_.predict(dtest)

    class XGBoostTrainerWrapper(BaseClassWrapper):
        """Wrap XGBoost's training process with nested callback support.

        This wrapper demonstrates:
        1. Wrapping an adapter class (XGBoostTrainer) that bridges procedural APIs
        2. Nested wrappers for callbacks with automatic parameter handling
        3. No need to override __init__/get_params/set_params - it all works automatically!
        4. Using _parameter_constraints to validate nested wrapper parameters
        5. Using _estimator_default_class to avoid passing the class every time
        """

        _estimator_name = "trainer"
        _estimator_base_class = object
        _estimator_default_class = XGBoostTrainer
        _parameter_constraints = {
            # Validates that callbacks is None or a BaseClassWrapper wrapping a class
            # that inherits from xgb.callback.TrainingCallback (the base for XGBoost callbacks)
            "callbacks": [None, {"wrapper_base_class": xgb.callback.TrainingCallback}]
        }

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y):
            self.instance_.fit_model(X, y)
            return self

        def predict(self, X):
            return self.instance_.predict_output(X)

    return XGBoostCallbackWrapper, XGBoostTrainer, XGBoostTrainerWrapper


@app.function(hide_code=True)
def create_slider(start, stop, value, label, step=None, **kwargs):
    params = {"start": start, "stop": stop, "value": value, "label": label, "show_value": True, **kwargs}
    if step is not None:
        params["step"] = step
    return mo.ui.slider(**params)


@app.cell(hide_code=True)
def _(create_slider, mo):
    depth_slider = create_slider(1, 10, 3, "Max Depth")
    eta_slider = create_slider(0.01, 0.5, 0.1, "Learning Rate (eta)", step=0.01)
    mo.hstack([depth_slider, eta_slider], justify="space-around")
    return depth_slider, eta_slider


@app.function(hide_code=True)
def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@app.cell
def _(
    XGBoostCallbackWrapper,
    XGBoostTrainerWrapper,
    depth_slider,
    eta_slider,
    xgb,
):
    # Create wrapped evaluation monitor callback (doesn't require validation set)
    eval_callback = XGBoostCallbackWrapper(
        callback=xgb.callback.EvaluationMonitor,
        period=10,
        show_stdv=False
    )

    # Use single callback (not list) to demonstrate nested parameter syntax
    wrapper = XGBoostTrainerWrapper(
        num_boost_round=50,
        callbacks=eval_callback,  # Single callback for nested params demo
        max_depth=depth_slider.value,
        eta=eta_slider.value,
        objective="reg:squarederror"
    )

    X_train, X_test, y_train, y_test = generate_regression_data(n_features=5, noise=20)
    wrapper.fit(X_train, y_train)

    y_pred_train = wrapper.predict(X_train)
    y_pred_test = wrapper.predict(X_test)
    X_plot = None
    y_pred_plot = None
    return wrapper, y_pred_test, y_pred_train, y_test, y_train


@app.function(hide_code=True)
def calculate_r2_score(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)


@app.function(hide_code=True)
def calculate_train_test_scores(y_train, y_pred_train, y_test, y_pred_test):
    return (calculate_r2_score(y_train, y_pred_train), calculate_r2_score(y_test, y_pred_test))


@app.cell(hide_code=True)
def _(
    depth_slider,
    eta_slider,
    mo,
    y_pred_test,
    y_pred_train,
    y_test,
    y_train,
):
    train_r2, test_r2 = calculate_train_test_scores(
        y_train, y_pred_train, y_test, y_pred_test
    )

    mo.md(
        f"""
        ## 3. Results

        **Train R²:** {train_r2:.3f}

        **Test R²:** {test_r2:.3f}

        Max Depth: {depth_slider.value}, Learning Rate: {eta_slider.value:.2f}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## HTML Representation

    Wrapped XGBoost models display correctly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### XGBoost Wrapper
    """)
    return


@app.cell
def _(wrapper):
    wrapper
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Nested Parameter Control

    Use the `__` syntax to control callback parameters.
    """)
    return


@app.cell
def _(wrapper):
    # Demonstrate nested parameter access
    params = wrapper.get_params(deep=True)

    # Show callback-related parameters
    callback_params = {k: str(v) for k, v in params.items() if 'callbacks' in k}
    callback_params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Modify Nested Parameters

    Change the callback period using nested syntax:
    """)
    return


@app.cell
def _(wrapper):
    # Use nested parameter syntax to change callback settings
    wrapper_modified = wrapper.set_params(callbacks__period=5)

    # Verify the change - automatic parameter handling works!
    modified_period = wrapper_modified.get_params(deep=True)['callbacks__period']
    modified_period
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - Low-level XGBoost API now sklearn-compatible
    - **Nested wrapper pattern** enables callback control via `__` syntax
    - Works with GridSearchCV for hyperparameter tuning (including callback params)
    - Can be used in sklearn Pipelines
    - Custom DMatrix handling integrated seamlessly
    - Clean HTML representation with nested structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [serialization.py](serialization.py) - Learn how to save and load wrapped XGBoost models with joblib
    """)
    return


if __name__ == "__main__":
    app.run()
