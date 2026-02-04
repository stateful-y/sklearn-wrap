"""
# Parameter Management

Master get_params() and set_params() for dynamic hyperparameter control.
Essential for GridSearchCV and interactive tuning.
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from sklearn_wrap import BaseClassWrapper
    return BaseClassWrapper, mo, np


@app.cell
def _(mo):
    mo.md("""
    ## Overview

    Master the `get_params()` and `set_params()` interface that makes sklearn's ecosystem work. These methods enable GridSearchCV to search hyperparameters, Pipeline to access nested parameters, and allow interactive parameter updates. Understanding this interface is essential for effective wrapper usage.
    """)
    return


@app.class_definition
class ConfigurableRegressor:
    def __init__(self, alpha=1.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        self.coef_ = self.alpha
        self.intercept_ = self.beta
        return self

    def predict(self, X):
        return X.flatten() * self.coef_ + self.intercept_


@app.cell
def _(BaseClassWrapper):
    class ConfigurableWrapper(BaseClassWrapper):
        _estimator_name = "model"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.fit(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return (ConfigurableWrapper,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Why It Matters

    These methods enable:
    - **GridSearchCV**: Iterates through parameter combinations
    - **Pipeline**: Accesses nested estimator parameters
    - **Cloning**: Creates copies with same parameters
    - **Introspection**: Discovers available parameters programmatically

    sklearn's entire ecosystem depends on this interface.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Interactive Parameter Control

    Sliders demonstrate real-time parameter updates.
    """)
    return
    def create_slider(start, stop, value, label, step=None, **kwargs):
        params = {"start": start, "stop": stop, "value": value, "label": label, "show_value": True, **kwargs}
        if step is not None:
            params["step"] = step
        return mo.ui.slider(**params)

    alpha_slider = create_slider(0.5, 50.0, 30.0, "Alpha (slope)", step=0.5)
    beta_slider = create_slider(-10.0, 10.0, 0.0, "Beta (intercept)", step=0.5)
    mo.hstack([alpha_slider, beta_slider], justify="space-around")
    return alpha_slider, beta_slider


@app.cell(hide_code=True)
def _(ConfigurableWrapper, alpha_slider, beta_slider, np):
    # Utility function for data generation
    def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create wrapper with slider values
    est = ConfigurableWrapper(
        estimator_class=ConfigurableRegressor,
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


@app.cell(hide_code=True)
def _(
    X_plot,
    X_test,
    X_train,
    alpha_slider,
    beta_slider,
    np,
    y_pred_plot,
    y_pred_test,
    y_pred_train,
    y_test,
    y_train,
):
    # Utility functions for visualization
    def calculate_r2_score(y_true, y_pred):
        return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)

    def calculate_train_test_scores(y_train, y_pred_train, y_test, y_pred_test):
        return (calculate_r2_score(y_train, y_pred_train), calculate_r2_score(y_test, y_pred_test))

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


@app.cell
def _(est, mo):
    params = est.get_params()
    mo.md(
        f"""
        ## 3. get_params() Deep Dive

        ```python
        {params}
        ```

        **What get_params() returns:**
        - All constructor parameters of the wrapped class
        - The `model` parameter (estimator_class itself)
        - With `deep=True` (default), includes nested estimator parameters

        **Key behaviors:**
        - Parameter names match the wrapped class's `__init__` signature
        - The `estimator_class` is read-only (cannot be changed via set_params)
        - Used by sklearn's `clone()` to create identical copies
        - GridSearchCV calls this to discover searchable parameters

        **Note:** The `model` key shows the class being wrapped, not the instance.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. set_params() Mechanics

    Update parameters dynamically without recreating the wrapper.
    """)
    return


@app.cell(hide_code=True)
def _(ConfigurableWrapper, X_test, X_train, y_train):
    # Create, update, and fit
    est2 = ConfigurableWrapper(estimator_class=ConfigurableRegressor, alpha=1.0, beta=0.0)

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


@app.cell
def _(error_msg, mo, updated_params):
    mo.md(f"""
    ### Updated Parameters

    ```python
    {updated_params}
    ```

    **How set_params() works:**
    1. Validates parameter names against wrapped class's `__init__`
    2. Updates internal `params` dictionary
    3. Returns `self` for method chaining
    4. Next `fit()` call uses updated parameters via `instantiate()`

    **Important:** Changes only take effect after calling `fit()` again.
    The wrapped instance is recreated during `instantiate()`.

    ### Invalid Parameter Error

    ```
    {error_msg}
    ```

    BaseClassWrapper validates against the wrapped class's `__init__` signature.
    Only parameters defined in `ConfigurableRegressor.__init__` are allowed.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [validation.py](validation.py) - Learn comprehensive error handling patterns and parameter constraint validation
    """)
    return


if __name__ == "__main__":
    app.run()
