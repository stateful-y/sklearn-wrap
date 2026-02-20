"""
# GridSearch with Wrappers

Use GridSearchCV to automatically find optimal hyperparameters for wrapped estimators.
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["numpy", "plotly", "scikit-learn", "sklearn-wrap"])
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    from sklearn.model_selection import GridSearchCV

    from sklearn_wrap import BaseClassWrapper

    return BaseClassWrapper, GridSearchCV, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What You'll Learn

    - How to use GridSearchCV with wrapped estimators for automated hyperparameter tuning
    - How BaseClassWrapper's parameter interface integrates seamlessly with sklearn's search tools
    - How to define parameter grids and interpret cross-validation results
    - How wrapped estimators display in sklearn meta-estimator HTML representations

    ## Prerequisites

    Familiarity with first_wrapper.py and parameter_interface.py.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. k-Nearest Neighbors Classifier

    A simple classifier to demonstrate GridSearchCV integration.
    """)
    return


@app.cell
def _(np):
    class KNNClassifier:
        """KNN classifier without sklearn conventions."""

        def __init__(self, n_neighbors=3, distance_metric="euclidean"):
            self._k_value = n_neighbors
            self._metric_type = distance_metric

        def train_classifier(self, X, y):
            """Store training data (not 'fit')."""
            self._training_features = X
            self._training_labels = y
            return self

        def classify(self, X):
            """Classify samples (not 'predict')."""
            predictions = []
            for x in X:
                distances = self._calculate_distances(x)
                nearest_indices = np.argsort(distances)[: self._k_value]
                nearest_labels = self._training_labels[nearest_indices]
                predictions.append(np.bincount(nearest_labels).argmax())
            return np.array(predictions)

        def _calculate_distances(self, x):
            if self._metric_type == "euclidean":
                return np.sqrt(((self._training_features - x) ** 2).sum(axis=1))
            elif self._metric_type == "manhattan":
                return np.abs(self._training_features - x).sum(axis=1)
            else:
                raise ValueError(f"Unknown metric: {self._metric_type}")

    return (KNNClassifier,)


@app.cell
def _(BaseClassWrapper):
    class KNNWrapper(BaseClassWrapper):
        _estimator_name = "knn"
        _estimator_base_class = object

        def fit(self, X, y):
            self.instantiate()
            self.instance_.train_classifier(X, y)

            # Mark estimator as fitted for sklearn compatibility
            self.fitted_ = True
            return self

        def predict(self, X):
            return self.instance_.classify(X)

    return (KNNWrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. GridSearchCV Execution
    """)
    return


@app.function(hide_code=True)
def generate_classification_data(n_samples=300, n_features=2, n_classes=2, test_size=0.3, random_state=42, **kwargs):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    # Ensure n_informative is sufficient for n_classes
    n_informative = max(n_features, n_classes)
    X, y = make_classification(n_samples=n_samples, n_features=n_informative, n_classes=n_classes, n_informative=n_informative, n_redundant=0, random_state=random_state, **kwargs)
    stratify = y if n_classes > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


@app.cell
def _(GridSearchCV, KNNClassifier, KNNWrapper):
    # Generate data
    X_train, X_test, y_train, y_test = generate_classification_data(n_samples=200, n_classes=3)

    # Define parameter grid
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "distance_metric": ["euclidean", "manhattan"],
    }

    # Create wrapper and run grid search
    wrapper = KNNWrapper(knn=KNNClassifier)
    grid_search = GridSearchCV(wrapper, param_grid, cv=3, scoring="accuracy", return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    cv_results = grid_search.cv_results_
    best_estimator = grid_search.best_estimator_
    return (
        best_estimator,
        best_params,
        best_score,
        cv_results,
        grid_search,
        test_score,
    )


@app.cell(hide_code=True)
def _(best_params, best_score, mo, test_score):
    mo.md(f"""
    ## 3. Results

    **Best Parameters:** `{best_params}`

    **Best CV Score:** {best_score:.3f}

    **Test Score:** {test_score:.3f}
    """)
    return


@app.function(hide_code=True)
def create_comparison_bars(categories, values_dict, title, yaxis_title="Score", colors=None, error_bars=None, **layout_kwargs):
    import plotly.graph_objects as go
    fig = go.Figure()
    default_colors = ["lightblue", "lightcoral", "lightgreen", "lightyellow"]
    for i, (name, values) in enumerate(values_dict.items()):
        color = colors.get(name) if colors else default_colors[i % len(default_colors)]
        trace_kwargs = {"name": name, "x": categories, "y": values, "marker": dict(color=color), "text": [f"{v:.4f}" for v in values], "textposition": "outside"}
        if error_bars and name in error_bars:
            trace_kwargs["error_y"] = dict(type="data", array=error_bars[name], visible=True)
        fig.add_trace(go.Bar(**trace_kwargs))
    fig.update_layout(title=title, yaxis_title=yaxis_title, barmode="group", height=400, **layout_kwargs)
    return fig


@app.cell
def _(cv_results, np):
    # Extract top 5 configurations
    sorted_indices = np.argsort(cv_results["rank_test_score"])[:5]

    categories = [
        f"k={cv_results['param_n_neighbors'][i]}, {cv_results['param_distance_metric'][i][:3]}"
        for i in sorted_indices
    ]

    values_dict = {
        "Train": [cv_results["mean_train_score"][i] for i in sorted_indices],
        "CV": [cv_results["mean_test_score"][i] for i in sorted_indices],
    }

    error_bars = {
        "CV": [cv_results["std_test_score"][i] for i in sorted_indices],
    }

    fig = create_comparison_bars(
        categories,
        values_dict,
        "Top 5 Configurations",
        yaxis_title="Accuracy",
        error_bars=error_bars,
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. HTML Representation

    GridSearchCV meta-estimators display correctly with wrapped estimators.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### GridSearchCV Object
    """)
    return


@app.cell
def _(grid_search):
    grid_search
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Best Estimator
    """)
    return


@app.cell
def _(best_estimator):
    best_estimator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **GridSearchCV** works seamlessly with wrapped estimators through the parameter interface
    - **Cross-validation** and scoring integrate automatically with no extra code in the wrapper
    - **Parameter validation** happens through BaseClassWrapper during each grid search iteration
    - **HTML representations** display correctly for meta-estimators containing wrappers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next Steps

    **Continue to:** [serialization.py](serialization.py) - Learn how to save and load wrapped estimators, pipelines, and GridSearchCV results
    """)
    return


if __name__ == "__main__":
    app.run()
