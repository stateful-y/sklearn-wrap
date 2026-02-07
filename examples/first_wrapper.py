"""
# Your First Wrapper

Create a minimal sklearn-compatible wrapper for any Python class using BaseClassWrapper.
Wrap a custom polynomial regression algorithm that doesn't follow sklearn conventions.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Imports
    import marimo as mo
    import numpy as np

    from sklearn_wrap import BaseClassWrapper

    return BaseClassWrapper, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Overview

    Learn how to wrap any Python class into a scikit-learn compatible estimator using `BaseClassWrapper`. This notebook demonstrates the core pattern by wrapping a custom polynomial regression algorithm that uses gradient descent and doesn't follow sklearn conventions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. The Pattern

    Any Python class can become sklearn-compatible in 3 steps:

    1. Inherit from `BaseClassWrapper`
    2. Set `_estimator_name` and `_estimator_base_class`
    3. Implement `fit()` and `predict()` (calling `instantiate()` first)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Custom Polynomial Regressor

    A non-sklearn class implementing polynomial regression with gradient descent.
    This class has its own method names and doesn't inherit from BaseEstimator.
    """)
    return


@app.cell
def _(np):
    class PolynomialRegressor:
        """Custom polynomial regressor without sklearn conventions.

        Uses train/compute_predictions instead of fit/predict.
        Stores parameters with different internal names.
        """

        def __init__(self, degree=1, learning_rate=0.01):
            # Internal attributes use different names than parameters
            self._poly_degree = degree
            self._step_size = learning_rate
            self._weights = None

        def train(self, X, y):
            """Train the model (not 'fit')."""
            X_poly = self._create_poly_features(X)
            n_samples, n_features = X_poly.shape
            self._weights = np.zeros(n_features)

            # Gradient descent
            for _ in range(1000):
                y_pred = X_poly @ self._weights
                gradient = -2 * X_poly.T @ (y - y_pred) / n_samples
                self._weights -= self._step_size * gradient

            return self

        def compute_predictions(self, X):
            """Compute predictions (not 'predict')."""
            if self._weights is None:
                raise ValueError("Must train before predict")
            X_poly = self._create_poly_features(X)
            return X_poly @ self._weights

        def _create_poly_features(self, X):
            X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
            features = [np.ones((X.shape[0], 1))]
            for d in range(1, self._poly_degree + 1):
                features.append(X**d)
            return np.hstack(features)

    return (PolynomialRegressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Wrap It for sklearn

    The wrapper bridges the custom class to sklearn's interface.
    """)
    return


@app.cell
def _(BaseClassWrapper):
    class PolyWrapper(BaseClassWrapper):
        _estimator_name = "poly_regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train(X, y)  # Call train(), not fit()

            # Mark estimator as fitted for sklearn compatibility
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.compute_predictions(X)  # Call compute_predictions()

    return (PolyWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Interactive Demo

    Adjust hyperparameters and see results in real-time.
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
    degree_slider = create_slider(1, 5, 2, "Polynomial Degree")
    lr_slider = create_slider(0.001, 0.1, 0.01, "Learning Rate", step=0.001)
    mo.hstack([degree_slider, lr_slider], justify="space-around")
    return degree_slider, lr_slider


@app.function(hide_code=True)
def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@app.cell
def _(PolyWrapper, PolynomialRegressor, degree_slider, lr_slider, np):
    # Create wrapper with slider values
    wrapper = PolyWrapper(
        estimator_class=PolynomialRegressor,
        degree=degree_slider.value,
        learning_rate=lr_slider.value,
    )

    X_train, X_test, y_train, y_test = generate_regression_data(n_features=1, noise=15)
    wrapper.fit(X_train, y_train)

    # Make predictions
    y_pred_train = wrapper.predict(X_train)
    y_pred_test = wrapper.predict(X_test)
    X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    y_pred_plot = wrapper.predict(X_plot)
    return (
        X_plot,
        X_test,
        X_train,
        wrapper,
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


@app.cell(hide_code=True)
def _(
    X_plot,
    X_test,
    X_train,
    calculate_train_test_scores,
    create_regression_scatter,
    degree_slider,
    lr_slider,
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
        title_prefix=f"Polynomial Regression (degree={degree_slider.value}, lr={lr_slider.value:.3f})",
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## HTML Representation

    Wrapped estimators display nicely in interactive environments (Jupyter, marimo, etc.).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Fitted Estimator
    """)
    return


@app.cell
def _(wrapper):
    wrapper
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - Custom algorithm now works with sklearn ecosystem
    - Works with `GridSearchCV` for hyperparameter tuning
    - Compatible with `Pipeline` for preprocessing chains
    - Supports `joblib` serialization
    - Automatic parameter validation
    - Clean HTML representation in interactive notebooks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [parameter_interface.py](parameter_interface.py) - Master the get_params/set_params interface for parameter management and sklearn integration
    """)
    return


if __name__ == "__main__":
    app.run()
