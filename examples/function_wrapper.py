# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "sklearn-wrap",
# ]
# ///
"""
# Function Wrapper

In this notebook, we wrap standalone functions into scikit-learn compatible
functors using FunctionWrapper.
"""

import marimo

__generated_with = "0.19.8"
__gallery__ = {
    "title": "Function Wrapper",
    "description": "Wrap standalone functions into sklearn-compatible functors with get_params/set_params using FunctionWrapper.",
    "category": "tutorial",
    "companion": "pages/tutorials/wrapping-functions.md",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    from sklearn.base import RegressorMixin, clone
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline

    from sklearn_wrap import FunctionWrapper
    from sklearn_wrap.base import _fit_context

    return (
        FunctionWrapper,
        GridSearchCV,
        Pipeline,
        RegressorMixin,
        _fit_context,
        clone,
        cross_val_score,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In this notebook, we wrap standalone functions into scikit-learn compatible
    functors using `FunctionWrapper`. The key convention: positional parameters
    are data (passed at call time), keyword-only parameters (after `*`) are
    config (stored on the wrapper and managed via `get_params`/`set_params`).

    **Prerequisites:** Basic familiarity with scikit-learn estimators.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. The Keyword-Only Convention

    `FunctionWrapper` identifies config parameters by looking at keyword-only
    arguments in your function signature. Everything before `*` is data,
    everything after is config.
    """)
    return


@app.cell
def _():
    def scaled_predict(X, *, scale=1.0, offset=0.0):
        """Positional args = data, keyword-only args = config."""
        return X.sum(axis=1) * scale + offset

    return (scaled_predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Basic Functor Usage

    Create a `FunctionWrapper` subclass by setting `_callable_name`. The wrapper
    automatically extracts keyword-only parameters and manages them via
    `get_params`/`set_params`.
    """)
    return


@app.cell
def _(FunctionWrapper):
    class ScaledPredictor(FunctionWrapper):
        _callable_name = "fn"

    return (ScaledPredictor,)


@app.cell
def _(ScaledPredictor, mo, scaled_predict):
    # Create a wrapper with custom config
    predictor = ScaledPredictor(fn=scaled_predict, scale=2.0, offset=1.0)

    # Inspect managed parameters
    mo.md(f"**Parameters:** `{predictor.get_params()}`")
    return (predictor,)


@app.cell
def _(mo, np, predictor):
    # Call it like a function: data goes in as positional args
    X_demo = np.array([[1, 2], [3, 4], [5, 6]])
    result = predictor(X_demo)
    mo.md(f"**Predictions:** `{result}`  (scale=2.0, offset=1.0)")
    return (X_demo,)


@app.cell
def _(clone, mo, predictor):
    # Clone preserves callable identity and config
    cloned = clone(predictor)
    mo.md(f"**Cloned params:** `{cloned.get_params()}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Full Estimator with Mixins

    By adding `RegressorMixin` and implementing `fit`/`predict` with
    `_fit_context`, you get a proper sklearn estimator that works in
    `Pipeline`, `GridSearchCV`, and `cross_val_score`.
    """)
    return


@app.cell
def _(FunctionWrapper, RegressorMixin, _fit_context, np):
    class FunctionRegressor(FunctionWrapper, RegressorMixin):
        _callable_name = "fn"

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            self.n_features_in_ = np.asarray(X).shape[1]
            return self

        def predict(self, X):
            X = np.asarray(X)
            return self.callable_fn(X, **self._params)

    return (FunctionRegressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Pipeline and GridSearchCV

    The wrapper integrates seamlessly with sklearn's pipeline and
    hyperparameter search tools.
    """)
    return


@app.cell
def _(np):
    # Generate sample data
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 3)
    y_train = X_train.sum(axis=1) * 2.0 + 1.0 + rng.randn(100) * 0.1
    return X_train, y_train


@app.cell
def _(FunctionRegressor, Pipeline, X_train, mo, scaled_predict, y_train):
    # Use in a Pipeline
    pipe = Pipeline([
        ("regressor", FunctionRegressor(fn=scaled_predict, scale=1.0)),
    ])
    pipe.fit(X_train, y_train)

    # Nested parameter access works
    mo.md(f"**Pipeline params:** scale={pipe.get_params()['regressor__scale']}, offset={pipe.get_params()['regressor__offset']}")
    return (pipe,)


@app.cell
def _(FunctionRegressor, GridSearchCV, X_train, mo, scaled_predict, y_train):
    # GridSearchCV over function config params
    grid = GridSearchCV(
        FunctionRegressor(fn=scaled_predict),
        param_grid={"scale": [0.5, 1.0, 2.0, 3.0], "offset": [0.0, 0.5, 1.0]},
        cv=3,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    mo.md(f"**Best params:** `{grid.best_params_}`  \n**Best score:** `{grid.best_score_:.4f}`")
    return (grid,)


@app.cell
def _(FunctionRegressor, X_train, cross_val_score, mo, scaled_predict, y_train):
    # Cross-validation
    scores = cross_val_score(
        FunctionRegressor(fn=scaled_predict, scale=2.0, offset=1.0),
        X_train, y_train,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    mo.md(f"**CV scores:** `{scores.round(4)}`  \n**Mean:** `{scores.mean():.4f}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. HTML Representation

    Wrapped functions display nicely in interactive environments, just like
    class-based estimators.
    """)
    return


@app.cell
def _(grid):
    grid.best_estimator_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What We Built

    We wrapped a standalone function into an sklearn-compatible functor using
    `FunctionWrapper`. Along the way, we:

    - Used the keyword-only convention to separate data from config parameters
    - Called the wrapper as a functor with `__call__`
    - Created a full estimator by adding `RegressorMixin` and `_fit_context`
    - Used the wrapper in `Pipeline` and `GridSearchCV`

    **Next steps:**

    - Wrapping classes instead of functions:
      [View](/examples/first_wrapper/) · [Open in marimo](/examples/first_wrapper/edit/)
    - The fit context decorator:
      [View](/examples/fit_context/) · [Open in marimo](/examples/fit_context/edit/)
    """)
    return


if __name__ == "__main__":
    app.run()
