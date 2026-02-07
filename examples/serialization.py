"""
# Serialization and Persistence

Save and load wrapped estimators, pipelines, and GridSearchCV objects using joblib.
All sklearn serialization mechanisms work seamlessly with wrapped estimators.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import numpy as np
    import tempfile
    from pathlib import Path

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    from sklearn_wrap import BaseClassWrapper

    return mo, np, tempfile, Path, GridSearchCV, Pipeline, StandardScaler, joblib, BaseClassWrapper


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Overview

    Save and load wrapped estimators using joblib, just like any sklearn estimator. Serialization preserves all state including fitted parameters, wrapped instances, and nested structures. This notebook demonstrates persistence for simple estimators, pipelines, and GridSearchCV meta-estimators.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## 1. Estimator Serialization

    Wrapped estimators can be saved and loaded like any sklearn estimator.
    """)
    return


@app.cell
def __(np):
    class SimpleRegressor:
        """A simple regressor without sklearn conventions."""

        def __init__(self, multiplier=1.0):
            self._scale_factor = multiplier

        def train_model(self, X, y):
            """Train by computing scaled mean (not 'fit')."""
            self._computed_value = y.mean() * self._scale_factor
            return self

        def generate_predictions(self, X):
            """Generate predictions (not 'predict')."""
            return np.full(X.shape[0], self._computed_value)
    return (SimpleRegressor,)


@app.cell
def __(BaseClassWrapper):
    class SimpleWrapper(BaseClassWrapper):
        _estimator_name = "regressor"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_model(X, y)
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.generate_predictions(X)
    return (SimpleWrapper,)


@app.function
def generate_regression_data(n_samples=100, n_features=1, noise=1.0, random_state=42):
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


@app.cell
def __(SimpleRegressor, SimpleWrapper, generate_regression_data):
    # Train and fit
    X_train, X_test, y_train, y_test = generate_regression_data()

    estimator = SimpleWrapper(estimator_class=SimpleRegressor, multiplier=2.0)
    estimator.fit(X_train, y_train)

    original_predictions = estimator.predict(X_test)
    original_score = np.mean((original_predictions - y_test) ** 2)
    return estimator, X_train, X_test, y_train, y_test, original_predictions, original_score


@app.cell
def __(estimator, joblib, tempfile):
    # Save to file
    temp_dir = tempfile.mkdtemp()
    estimator_path = f"{temp_dir}/estimator.pkl"
    joblib.dump(estimator, estimator_path)

    # Load from file
    loaded_estimator = joblib.load(estimator_path)
    return temp_dir, estimator_path, loaded_estimator


@app.cell
def __(loaded_estimator, X_test, y_test, original_predictions, original_score, mo, np):
    # Verify loaded estimator works
    loaded_predictions = loaded_estimator.predict(X_test)
    loaded_score = np.mean((loaded_predictions - y_test) ** 2)

    mo.md(f"""
    ### Estimator Serialization

    - Original MSE: {original_score:.2f}
    - Loaded MSE: {loaded_score:.2f}
    - Predictions match: {np.allclose(original_predictions, loaded_predictions)}

    The loaded estimator produces identical predictions.
    """)
    return loaded_predictions, loaded_score


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## 2. Pipeline Serialization

    Pipelines containing wrapped estimators serialize correctly.
    """)
    return


@app.cell
def __(SimpleRegressor, SimpleWrapper, StandardScaler, Pipeline, X_test, X_train, y_train):
    # Create and fit pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", SimpleWrapper(estimator_class=SimpleRegressor, multiplier=1.5))
    ])

    pipeline.fit(X_train, y_train)
    pipeline_predictions = pipeline.predict(X_test)
    return pipeline, pipeline_predictions


@app.cell
def __(pipeline, joblib, temp_dir):
    # Save and load pipeline
    pipeline_path = f"{temp_dir}/pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    loaded_pipeline = joblib.load(pipeline_path)
    return pipeline_path, loaded_pipeline


@app.cell
def __(loaded_pipeline, X_test):
    # Verify pipeline
    loaded_pipeline_predictions = loaded_pipeline.predict(X_test)
    return loaded_pipeline_predictions


@app.cell(hide_code=True)
def __(mo, np, pipeline_predictions, loaded_pipeline_predictions, loaded_pipeline):
    mo.md(f"""
    ### Pipeline Serialization

    - Predictions match: {np.allclose(pipeline_predictions, loaded_pipeline_predictions)}
    - Pipeline steps preserved: {list(loaded_pipeline.named_steps.keys())}

    The loaded pipeline maintains all preprocessing steps and wrapped estimator.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## 3. GridSearchCV Serialization

    Meta-estimators save complete search results and best parameters.
    """)
    return


@app.cell
def __(SimpleRegressor, SimpleWrapper, GridSearchCV, X_test, X_train, y_train):
    # Run grid search
    grid = GridSearchCV(
        SimpleWrapper(estimator_class=SimpleRegressor),
        param_grid={"multiplier": [0.5, 1.0, 1.5, 2.0]},
        cv=3,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train, y_train)
    grid_predictions = grid.predict(X_test)
    best_params = grid.best_params_
    return grid, grid_predictions, best_params


@app.cell(hide_code=True)
def __(grid, joblib, temp_dir):
    # Save and load grid search
    grid_path = f"{temp_dir}/grid.pkl"
    joblib.dump(grid, grid_path)
    loaded_grid = joblib.load(grid_path)
    return grid_path, loaded_grid


@app.cell
def __(loaded_grid, X_test, grid_predictions, best_params, mo, np):
    # Verify grid search
    loaded_grid_predictions = loaded_grid.predict(X_test)
    return loaded_grid_predictions


@app.cell(hide_code=True)
def __(loaded_grid, best_params, grid_predictions, loaded_grid_predictions,  mo, np):
    mo.md(f"""
    ### GridSearchCV Serialization

    - Best params preserved: `{loaded_grid.best_params_}`
    - Original best params: `{best_params}`
    - Predictions match: {np.allclose(grid_predictions, loaded_grid_predictions)}
    - CV results available: {len(loaded_grid.cv_results_['params'])} configs tested

    The loaded GridSearchCV retains best estimator and all cross-validation results.
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Key Takeaways

    - Wrapped estimators serialize with `joblib.dump()` and `pickle`
    - Pipelines containing wrappers persist correctly
    - Meta-estimators like GridSearchCV save all state
    """)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [xgboost_wrapper.py](xgboost_wrapper.py) - Learn to wrap third-party libraries like XGBoost
    """)
    return


if __name__ == "__main__":
    app.run()
