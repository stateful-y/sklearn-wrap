"""
# Nested Parameters

Master the `__` syntax for controlling parameters in nested estimator hierarchies.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    from sklearn_wrap import BaseClassWrapper

    return BaseClassWrapper, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Overview

    Master the double-underscore (`__`) syntax for controlling parameters in nested estimator hierarchies. When wrappers contain other wrappers, sklearn's parameter interface uses `outer__inner__param` notation to access nested parameters. This enables GridSearchCV to search complex parameter spaces and allows precise control over deeply nested structures.
    """)
    return


@app.cell
def _(np):
    class BaseRegressor:
        """Non-sklearn regressor with custom methods."""

        def __init__(self, scale=1.0):
            self._scaling_factor = scale

        def train_model(self, X, y):
            """Train the model (not 'fit')."""
            self._average_value = y.mean()
            return self

        def generate_output(self, X):
            """Generate predictions (not 'predict')."""
            return np.full(X.shape[0], self._average_value * self._scaling_factor)


    class EnsembleRegressor:
        """Ensemble that doesn't follow sklearn conventions."""

        def __init__(self, estimator1, estimator2, blend=0.5):
            self._model_a = estimator1
            self._model_b = estimator2
            self._blend_ratio = blend

        def train_ensemble(self, X, y):
            """Train both models (not 'fit')."""
            self._model_a.fit(X, y)
            self._model_b.fit(X, y)
            return self

        def compute_blend(self, X):
            """Compute blended predictions (not 'predict')."""
            pred1 = self._model_a.predict(X)
            pred2 = self._model_b.predict(X)
            return self._blend_ratio * pred1 + (1 - self._blend_ratio) * pred2

    return (BaseRegressor, EnsembleRegressor)


@app.cell
def _(BaseClassWrapper):
    class BaseWrapper(BaseClassWrapper):
        _estimator_name = "regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_model(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.generate_output(X)

    return (BaseWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Create Nested Estimator
    """)
    return


@app.cell
def _(BaseClassWrapper, BaseRegressor, BaseWrapper, EnsembleRegressor):
    class EnsembleWrapper(BaseClassWrapper):
        _estimator_name = "ensemble"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_ensemble(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.compute_blend(X)

    # Create two inner estimators with different scales
    inner1 = BaseWrapper(estimator_class=BaseRegressor, scale=0.8)
    inner2 = BaseWrapper(estimator_class=BaseRegressor, scale=1.2)

    # Create ensemble with nested estimators
    ensemble = EnsembleWrapper(
        estimator_class=EnsembleRegressor,
        estimator1=inner1,
        estimator2=inner2,
        blend=0.5
    )

    return (EnsembleWrapper, ensemble, inner1, inner2)


@app.cell(hide_code=True)
def _(ensemble, mo):
    params = ensemble.get_params(deep=True)

    # Filter to show nested params clearly
    nested_params = {k: v for k, v in params.items() if '__' in k}

    mo.md(
        f"""
        ## 2. Parameter Inspection

        **All parameters:**
        ```python
        {list(params.keys())}
        ```

        **Nested parameters (with __):**
        ```python
        {nested_params}
        ```

        Notice the `__` syntax for nested access like `estimator1__scale`.
        """
    )
    return (nested_params, params)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Interactive Control
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
    scale1_slider = create_slider(0.5, 1.5, 0.8, "Scale 1", step=0.1)
    scale2_slider = create_slider(0.5, 1.5, 1.2, "Scale 2", step=0.1)
    blend_slider = create_slider(0.0, 1.0, 0.5, "Blend", step=0.1)
    mo.hstack([scale1_slider, scale2_slider, blend_slider], justify="space-around")
    return blend_slider, scale1_slider, scale2_slider


@app.cell
def _(BaseRegressor, BaseWrapper, EnsembleRegressor, EnsembleWrapper, blend_slider, scale1_slider, scale2_slider):
    # Create ensemble with slider-controlled parameters using nested syntax
    ensemble_interactive = EnsembleWrapper(
        estimator_class=EnsembleRegressor,
        estimator1=BaseWrapper(estimator_class=BaseRegressor, scale=scale1_slider.value),
        estimator2=BaseWrapper(estimator_class=BaseRegressor, scale=scale2_slider.value),
        blend=blend_slider.value
    )

    # Generate and fit data
    X_train, X_test, y_train, y_test = generate_regression_data(n_features=1, noise=10)
    ensemble_interactive.fit(X_train, y_train)

    # Make predictions
    y_pred_train = ensemble_interactive.predict(X_train)
    y_pred_test = ensemble_interactive.predict(X_test)

    # For plotting
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_pred_plot = ensemble_interactive.predict(X_plot)

    return (ensemble_interactive, X_plot, X_test, X_train, y_pred_plot, y_pred_test, y_pred_train, y_test, y_train)


@app.cell
def _(
    X_plot,
    X_test,
    X_train,
    blend_slider,
    calculate_train_test_scores,
    create_regression_scatter,
    mo,
    scale1_slider,
    scale2_slider,
    y_pred_plot,
    y_pred_test,
    y_pred_train,
    y_test,
    y_train,
):
    train_r2, test_r2 = calculate_train_test_scores(
        y_train, y_pred_train, y_test, y_pred_test
    )

    fig = create_regression_scatter(
        X_train, y_train, X_test, y_test, X_plot, y_pred_plot,
        train_r2, test_r2,
        title_prefix=f"Ensemble: scale1={scale1_slider.value:.1f}, scale2={scale2_slider.value:.1f}, blend={blend_slider.value:.1f}"
    )

    mo.ui.plotly(fig)
    return fig, train_r2, test_r2


@app.function
def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Nested Parameter Modification

    Use `set_params` with `__` syntax to modify nested parameters without recreating the estimator.
    """)
    return


@app.cell
def _(ensemble):
    # Demonstrate nested parameter modification
    ensemble_modified = ensemble.set_params(
        estimator1__scale=1.5,
        estimator2__scale=0.7,
        blend=0.3
    )

    # Verify changes
    modified_params = {
        'estimator1__scale': ensemble_modified.get_params(deep=True)['estimator1__scale'],
        'estimator2__scale': ensemble_modified.get_params(deep=True)['estimator2__scale'],
        'blend': ensemble_modified.get_params(deep=True)['blend']
    }

    modified_params
    return (ensemble_modified, modified_params)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. HTML Representation

    The nested estimator structure displays correctly with all parameter hierarchies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ensemble Estimator
    """)
    return


@app.cell
def _(ensemble):
    ensemble
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [fit_context.py](fit_context.py) - Understand the `_fit_context` decorator and automatic validation control
    """)
    return


if __name__ == "__main__":
    app.run()
