"""
# XGBoost Integration

Wrap XGBoost's low-level Booster API for sklearn compatibility.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np

    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        xgb = None

    from sklearn_wrap import BaseClassWrapper

    return mo, np, xgb, XGBOOST_AVAILABLE, BaseClassWrapper


@app.cell
def __(mo):
    mo.md("""
    ## Overview

    Wrap XGBoost's low-level Booster API to make it sklearn-compatible. XGBoost's native API uses DMatrix objects and doesn't follow sklearn conventions. This notebook shows how to bridge the gap, enabling XGBoost to work with GridSearchCV, Pipelines, and other sklearn tools.
    """)
    return


@app.cell
def __(mo, XGBOOST_AVAILABLE):
    if not XGBOOST_AVAILABLE:
        mo.md(
            """
            ## XGBoost Not Available

            Install xgboost to run this example:
            ```
            uv add --group examples xgboost
            ```
            """
        )
    return


@app.cell
def __(XGBOOST_AVAILABLE, BaseClassWrapper, xgb):
    if XGBOOST_AVAILABLE:
        mo.md("""
            ## 1. XGBoost Wrapper Setup

            Wrap the low-level Booster API with custom DMatrix handling.
            """)

        class XGBoostBoosterWrapper(BaseClassWrapper):
            """Wrap XGBoost's low-level Booster API."""

            _estimator_name = "booster"
            _estimator_base_class = object

            def __init__(self, estimator_class, num_boost_round=100, **params):
                # XGBoost Booster expects DMatrix, not raw arrays
                self.num_boost_round = num_boost_round
                super().__init__(estimator_class, **params)

            def fit(self, X, y):
                dtrain = xgb.DMatrix(X, label=y)
                self.instance_ = xgb.train(
                    self.params, dtrain, num_boost_round=self.num_boost_round
                )
                self.fitted_ = True
                return self

            def predict(self, X):
                dtest = xgb.DMatrix(X)
                return self.instance_.predict(dtest)

            def instantiate(self):
                # Override: Booster is created during fit, not before
                return self
    else:
        XGBoostBoosterWrapper = None

    return (XGBoostBoosterWrapper,)


@app.cell
def __(mo, XGBOOST_AVAILABLE):
    if XGBOOST_AVAILABLE:
        mo.md("## 2. Interactive Tuning")
    return


@app.cell(hide_code=True)
def __(XGBOOST_AVAILABLE, mo):
    if XGBOOST_AVAILABLE:
        # Utility function for creating sliders
        def create_slider(start, stop, value, label, step=None, **kwargs):
            params = {"start": start, "stop": stop, "value": value, "label": label, "show_value": True, **kwargs}
            if step is not None:
                params["step"] = step
            return mo.ui.slider(**params)

        depth_slider = create_slider(1, 10, 3, "Max Depth")
        eta_slider = create_slider(0.01, 0.5, 0.1, "Learning Rate (eta)", step=0.01)
        mo.hstack([depth_slider, eta_slider], justify="space-around")
    else:
        depth_slider = None
        eta_slider = None
        create_slider = None
        mo.md("")
    return depth_slider, eta_slider, create_slider


@app.cell(hide_code=True)
def __(
    XGBOOST_AVAILABLE,
    XGBoostBoosterWrapper,
    xgb,
    depth_slider,
    eta_slider,
):
    if XGBOOST_AVAILABLE:
        # Utility function for data generation
        def generate_regression_data(n_samples=300, n_features=2, noise=20, test_size=0.3, random_state=42, **kwargs):
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state, **kwargs)
            return train_test_split(X, y, test_size=test_size, random_state=random_state)

        wrapper = XGBoostBoosterWrapper(
            estimator_class=xgb.Booster,
            num_boost_round=50,
            max_depth=depth_slider.value,
            eta=eta_slider.value,
            objective="reg:squarederror",
        )

        X_train, X_test, y_train, y_test = generate_regression_data(n_features=5, noise=20)
        wrapper.fit(X_train, y_train)

        y_pred_train = wrapper.predict(X_train)
        y_pred_test = wrapper.predict(X_test)
        X_plot = None
        y_pred_plot = None
    else:
        generate_regression_data = None
        wrapper = None
        X_train = X_test = y_train = y_test = None
        y_pred_train = y_pred_test = None
        X_plot = y_pred_plot = None

    return wrapper, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, X_plot, y_pred_plot, generate_regression_data


@app.cell(hide_code=True)
def __(
    XGBOOST_AVAILABLE,
    mo,
    np,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    depth_slider,
    eta_slider,
):
    if XGBOOST_AVAILABLE:
        # Utility functions for calculating scores
        def calculate_r2_score(y_true, y_pred):
            return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)

        def calculate_train_test_scores(y_train, y_pred_train, y_test, y_pred_test):
            return (calculate_r2_score(y_train, y_pred_train), calculate_r2_score(y_test, y_pred_test))

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
    else:
        train_r2 = test_r2 = None
        mo.md("")
    return train_r2, test_r2


@app.cell
def __(mo, XGBOOST_AVAILABLE):
    if XGBOOST_AVAILABLE:
        mo.md("""
            ## HTML Representation

            Wrapped XGBoost models display correctly.
            """)
    return


@app.cell
def __(wrapper, mo, XGBOOST_AVAILABLE):
    if XGBOOST_AVAILABLE:
        mo.md("### XGBoost Wrapper")
        wrapper
    return


@app.cell
def __(mo, XGBOOST_AVAILABLE):
    if XGBOOST_AVAILABLE:
        mo.md(
            """
            ## Key Takeaways

            - Low-level XGBoost API now sklearn-compatible
            - Works with GridSearchCV for hyperparameter tuning
            - Can be used in sklearn Pipelines
            - Custom DMatrix handling integrated seamlessly
            - Clean HTML representation
            """
        )
    return


@app.cell
def __(mo, XGBOOST_AVAILABLE):
    if XGBOOST_AVAILABLE:
        mo.md(
            """
            ## Next Steps

            **Continue to:** [serialization.py](serialization.py) - Learn how to save and load wrapped XGBoost models with joblib
            """
        )
    return


if __name__ == "__main__":
    app.run()
