"""
# Nested Parameters

Master the `__` syntax for controlling parameters in nested estimator hierarchies.
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

    Master the double-underscore (`__`) syntax for controlling parameters in nested estimator hierarchies. When wrappers contain other wrappers, sklearn's parameter interface uses `outer__inner__param` notation to access nested parameters. This enables GridSearchCV to search complex parameter spaces and allows precise control over deeply nested structures.
    """)
    return


@app.cell
def _(np):
    class BaseRegressor:
        def __init__(self, scale=1.0):
            self.scale = scale

        def fit(self, X, y):
            self.mean_ = y.mean()
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.mean_ * self.scale)
    return (BaseRegressor,)


@app.class_definition
class EnsembleRegressor:
    def __init__(self, estimator1, estimator2, blend=0.5):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.blend = blend

    def fit(self, X, y):
        self.estimator1.fit(X, y)
        self.estimator2.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.estimator1.predict(X)
        pred2 = self.estimator2.predict(X)
        return self.blend * pred1 + (1 - self.blend) * pred2


@app.cell
def _(BaseClassWrapper):
    class BaseWrapper(BaseClassWrapper):
        _estimator_name = "regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.fit(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return (BaseWrapper,)


@app.cell
def _(BaseClassWrapper):
    class EnsembleWrapper(BaseClassWrapper):
        _estimator_name = "ensemble"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.fit(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.predict(X)
    return (EnsembleWrapper,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Create Nested Estimator
    """)
    return


@app.cell
def _(BaseRegressor, BaseWrapper, EnsembleWrapper):
    # Create nested structure: EnsembleWrapper -> EnsembleRegressor -> 2x BaseWrapper
    wrapper1 = BaseWrapper(estimator_class=BaseRegressor, scale=0.8)
    wrapper2 = BaseWrapper(estimator_class=BaseRegressor, scale=1.2)

    ensemble = EnsembleWrapper(
        estimator_class=EnsembleRegressor,
        estimator1=wrapper1,
        estimator2=wrapper2,
        blend=0.5,
    )
    return (ensemble,)


@app.cell
def _(ensemble, mo):
    params = ensemble.get_params(deep=True)
    mo.md(
        f"""
        ## 2. Parameter Inspection

        ```python
        {params}
        ```

        Notice the `__` syntax for nested access.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Interactive Control
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Utility function for creating sliders
    def create_slider(start, stop, value, label, step=None, **kwargs):
        params = {"start": start, "stop": stop, "value": value, "label": label, "show_value": True, **kwargs}
        if step is not None:
            params["step"] = step
        return mo.ui.slider(**params)

    scale1_slider = create_slider(0.5, 1.5, 0.8, "Scale 1", step=0.1)
    scale2_slider = create_slider(0.5, 1.5, 1.2, "Scale 2", step=0.1)
    blend_slider = create_slider(0.0, 1.0, 0.5, "Blend", step=0.1)
    mo.hstack([scale1_slider, scale2_slider, blend_slider], justify="space-around")
    return blend_slider, scale1_slider, scale2_slider


@app.cell(hide_code=True)
def _(blend_slider, ensemble, np, scale1_slider, scale2_slider):
    # Utility function for data generation
    def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Update nested parameters using __ syntax
    ensemble.set_params(
        estimator1__scale=scale1_slider.value,
        estimator2__scale=scale2_slider.value,
        blend=blend_slider.value,
    )

    X_train, X_test, y_train, y_test = generate_regression_data(n_features=1)
    ensemble.fit(X_train, y_train)

    y_pred_train = ensemble.predict(X_train)
    y_pred_test = ensemble.predict(X_test)
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_pred_plot = ensemble.predict(X_plot)
    return (
        X_plot,
        X_test,
        X_train,
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
    blend_slider,
    np,
    scale1_slider,
    scale2_slider,
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
        title_prefix=f"Ensemble (s1={scale1_slider.value:.1f}, s2={scale2_slider.value:.1f}, blend={blend_slider.value:.1f})",
    )
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## HTML Representation

    The nested estimator structure displays correctly with all parameter hierarchies.
    """)
    return


@app.cell
def _(ensemble, mo):
    mo.md("### Ensemble Estimator")
    ensemble
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Takeaways

    - Use `estimator__param` to access nested parameters
    - Works with any depth of nesting
    - GridSearchCV can search nested parameter spaces
    - `get_params(deep=True)` shows full hierarchy
    - HTML representation shows complete nested structure
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [fit_context.py](fit_context.py) - Understand the `_fit_context` decorator and automatic validation control
    """)
    return


if __name__ == "__main__":
    app.run()
