# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "plotly",
#     "scikit-learn",
#     "sklearn-wrap",
# ]
# ///
"""
# Parameter Management

We explore how `get_params()` and `set_params()` work in wrapped estimators,
and why sklearn's ecosystem depends on this interface.
"""

import marimo

__generated_with = "0.19.8"
__gallery__ = {
    "title": "The Parameter Interface",
    "description": "Explore get_params and set_params for GridSearchCV and Pipeline integration.",
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
    return BaseClassWrapper, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In this tutorial we examine how `BaseClassWrapper` implements sklearn's parameter
    interface. We will use interactive sliders to see parameter changes in real time,
    inspect `get_params()` output, and experiment with `set_params()` - including what
    happens with invalid parameters.

    **Prerequisites** - Familiarity with [first_wrapper.py](first_wrapper.py).
    """)
    return


@app.cell(hide_code=True)
def _():
    class ConfigurableRegressor:
        """Non-sklearn regressor with different method names."""

        def __init__(self, alpha=1.0, beta=0.0):
            # Store with different internal names
            self._slope = alpha
            self._offset = beta

        def train_model(self, X, y):
            """Train the model (not 'fit')."""
            self._coefficient = self._slope
            self._intercept_value = self._offset
            return self

        def make_predictions(self, X):
            """Make predictions (not 'predict')."""
            return X.flatten() * self._coefficient + self._intercept_value
    return (ConfigurableRegressor,)


@app.cell
def _(BaseClassWrapper):
    class ConfigurableWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        def fit(self, X, y):
            """sklearn fit that delegates to train_model()."""
            self.instantiate()
            self.instance_.train_model(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            """sklearn predict that delegates to make_predictions()."""
            return self.instance_.make_predictions(X)
    return (ConfigurableWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Why Parameters Matter

    `get_params()` and `set_params()` are the interface that `GridSearchCV`, `Pipeline`,
    and `clone()` all rely on. `BaseClassWrapper` implements them automatically by
    inspecting the wrapped class's `__init__` signature.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Interactive Parameter Control

    Let's use sliders to change `alpha` and `beta` and watch the model update in real time.
    """)
    return


@app.function(hide_code=True)
def create_slider(start, stop, value, label, step=None, **kwargs):
    params = {"start": start, "stop": stop, "value": value, "label": label, "show_value": True, **kwargs}
    if step is not None:
        params["step"] = step
    return mo.ui.slider(**params)


@app.cell(hide_code=True)
def _(create_slider, mo):
    alpha_slider = create_slider(0.5, 50.0, 30.0, "Alpha (slope)", step=0.5)
    beta_slider = create_slider(-10.0, 10.0, 0.0, "Beta (intercept)", step=0.5)
    mo.hstack([alpha_slider, beta_slider], justify="space-around")
    return alpha_slider, beta_slider


@app.function(hide_code=True)
def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@app.cell
def _(ConfigurableRegressor, ConfigurableWrapper, alpha_slider, beta_slider, generate_regression_data, np):
    # Create wrapper with slider values
    est = ConfigurableWrapper(
        model=ConfigurableRegressor,
        alpha=alpha_slider.value,
        beta=beta_slider.value,
    )

    X_train, X_test, y_train, y_test = generate_regression_data(n_features=1, noise=10)
    est.fit(X_train, y_train)

    y_pred_train = est.predict(X_train)
    y_pred_test = est.predict(X_test)
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_pred_plot = est.predict(X_plot)
    return (
        X_plot,
        X_test,
        X_train,
        est,
        y_pred_plot,
        y_pred_test,
        y_pred_train,
        y_test,
        y_train,
    )


@app.function(hide_code=True)
def calculate_r2_score(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)


@app.function(hide_code=True)
def calculate_train_test_scores(y_train, y_pred_train, y_test, y_pred_test):
    return (calculate_r2_score(y_train, y_pred_train), calculate_r2_score(y_test, y_pred_test))


@app.function(hide_code=True)
def create_regression_scatter(X_train, y_train, X_test, y_test, X_plot, y_pred_plot, train_score, test_score, title_prefix="", **layout_kwargs):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train, mode="markers", name="Training Data", marker=dict(size=8, color="lightblue", line=dict(width=1, color="darkblue"))))
    fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode="markers", name="Test Data", marker=dict(size=8, color="lightcoral", line=dict(width=1, color="darkred"))))
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_pred_plot, mode="lines", name="Model Prediction", line=dict(color="green", width=3)))
    title = f"Train R² = {train_score:.3f}, Test R² = {test_score:.3f}"
    if title_prefix:
        title = f"{title_prefix}<br>{title}"
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Target", height=500, showlegend=True, **layout_kwargs)
    return fig


@app.cell
def _(
    X_plot,
    X_test,
    X_train,
    alpha_slider,
    beta_slider,
    calculate_train_test_scores,
    create_regression_scatter,
    y_pred_plot,
    y_pred_test,
    y_pred_train,
    y_test,
    y_train,
):
    train_r2, test_r2 = calculate_train_test_scores(y_train, y_pred_train, y_test, y_pred_test)

    fig = create_regression_scatter(
        X_train,
        y_train,
        X_test,
        y_test,
        X_plot,
        y_pred_plot,
        train_r2,
        test_r2,
        title_prefix=f"α={alpha_slider.value:.1f}, β={beta_slider.value:.1f}",
    )
    fig
    return


@app.cell(hide_code=True)
def _(est, mo):
    params = est.get_params()
    mo.md(
        f"""
        ## 3. Inspecting get_params()

        ```python
        {params}
        ```

        Notice that `get_params()` returns every `__init__` parameter of the wrapped class
        plus the `model` key (the class itself). This is exactly what `GridSearchCV` and
        `clone()` use to discover and reproduce an estimator's configuration.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Updating with set_params()

    We can update parameters dynamically without recreating the wrapper.
    """)
    return


@app.cell
def _(ConfigurableWrapper, ConfigurableRegressor, X_test, X_train, y_train):
    # Create, update, and fit
    est2 = ConfigurableWrapper(model=ConfigurableRegressor, alpha=1.0, beta=0.0)

    # set_params() returns self for method chaining
    est2.set_params(alpha=2.5, beta=-1.0)
    est2.fit(X_train, y_train)

    updated_params = est2.get_params()
    y_pred_updated = est2.predict(X_test)

    # Error demo: invalid parameter
    error_msg = None
    try:
        est2.set_params(invalid_param=999)
    except ValueError as e:
        error_msg = str(e)
    return error_msg, updated_params


@app.cell(hide_code=True)
def _(error_msg, mo, updated_params):
    mo.md(f"""
    ### Updated Parameters

    ```python
    {updated_params}
    ```

    Notice that `set_params()` validates names against the wrapped class's `__init__`,
    updates the internal `params` dictionary, and returns `self` for method chaining.
    Changes take effect on the next `fit()` call, when the wrapped instance is recreated.

    ### Invalid Parameter Error

    ```
    {error_msg}
    ```

    Passing a parameter that does not exist in `ConfigurableRegressor.__init__` raises
    a `ValueError` immediately - we never reach `fit()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What We Built

    We used `get_params()` and `set_params()` to inspect and update a wrapped estimator's
    configuration at runtime. The parameter interface is what makes `BaseClassWrapper`
    compatible with `GridSearchCV`, `Pipeline`, and `clone()`.

    Next: [grid_search.py](grid_search.py) puts this interface to work inside `GridSearchCV`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **More examples:** [validation.py](validation.py) | [nested_wrappers.py](nested_wrappers.py)
    """)
    return


if __name__ == "__main__":
    app.run()
