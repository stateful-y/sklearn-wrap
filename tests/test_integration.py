"""Integration tests for BaseClassWrapper with sklearn tools.

Tests integration with:
- Pipeline
- GridSearchCV
- cross_val_score
- StackingRegressor/StackingClassifier
- VotingClassifier/VotingRegressor

These tests verify that wrappers work correctly with sklearn's
ecosystem of tools and utilities.
"""

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import StackingRegressor, VotingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_wrap.base import BaseClassWrapper, _fit_context

from .conftest import BaseTestClass

# ============================================================================
# Test Estimator Classes
# ============================================================================


class SimpleRegressorClass(BaseTestClass):
    """Simple regressor for testing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.coef_ = np.ones(X.shape[1]) * self.alpha
        return self

    def predict(self, X):
        return X @ self.coef_


class SimpleClassifierClass(BaseTestClass):
    """Simple classifier for testing."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mean_ = np.mean(X, axis=0)
        return self

    def predict(self, X):
        # Simple prediction based on distance to mean
        distances = np.sum((X - self.mean_) ** 2, axis=1)
        return (distances > self.threshold).astype(int)


# ============================================================================
# Wrapper Classes
# ============================================================================


class RegressorWrapper(RegressorMixin, BaseClassWrapper):
    """Wrapper for regression estimators."""

    _estimator_name = "regressor"
    _estimator_base_class = BaseTestClass

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)

    def score(self, X, y):
        """Return R^2 score."""
        return r2_score(y, self.predict(X))


class ClassifierWrapper(ClassifierMixin, BaseClassWrapper):
    """Wrapper for classification estimators."""

    _estimator_name = "classifier"
    _estimator_base_class = BaseTestClass

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self.instance_.fit(X, y)
        return self

    def predict(self, X):
        return self.instance_.predict(X)

    def score(self, X, y):
        """Return accuracy score."""
        return accuracy_score(y, self.predict(X))


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


@pytest.mark.integration
class TestPipelineIntegration:
    """Test integration with sklearn Pipeline."""

    def test_wrapper_in_pipeline(self):
        """Test that wrapper works as a step in Pipeline."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)

        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", wrapper)])

        # Should fit without error
        pipeline.fit(X, y)

        # Should predict
        predictions = pipeline.predict(X)
        assert predictions.shape == (100,)

    def test_pipeline_with_nested_params(self):
        """Test setting nested parameters in Pipeline."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", wrapper)])

        # Set nested parameter
        pipeline.set_params(regressor__alpha=2.0)
        assert pipeline.named_steps["regressor"].params["alpha"] == 2.0

        # Fit and verify it works
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (100,)


# ============================================================================
# GridSearchCV Integration Tests
# ============================================================================


@pytest.mark.integration
class TestGridSearchCVIntegration:
    """Test integration with sklearn GridSearchCV."""

    def test_gridsearchcv_with_wrapper(self):
        """Test GridSearchCV with wrapped estimator."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)

        param_grid = {"alpha": [0.5, 1.0, 2.0]}

        grid_search = GridSearchCV(wrapper, param_grid, cv=3)
        grid_search.fit(X, y)

        # Should have best_params_ and best_estimator_
        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_estimator_")
        assert "alpha" in grid_search.best_params_

    def test_gridsearchcv_in_pipeline(self):
        """Test GridSearchCV with wrapper in Pipeline."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", wrapper)])

        param_grid = {"regressor__alpha": [0.5, 1.0, 2.0]}

        grid_search = GridSearchCV(pipeline, param_grid, cv=3)
        grid_search.fit(X, y)

        assert "regressor__alpha" in grid_search.best_params_

    def test_gridsearchcv_preserves_best_estimator(self):
        """Test that GridSearchCV preserves the best estimator correctly."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)

        param_grid = {"alpha": [0.5, 1.0, 2.0, 5.0]}

        grid_search = GridSearchCV(wrapper, param_grid, cv=3)
        grid_search.fit(X, y)

        # Best estimator should be fitted and able to predict
        predictions = grid_search.best_estimator_.predict(X)
        assert predictions.shape == (100,)


# ============================================================================
# cross_val_score Integration Tests
# ============================================================================


@pytest.mark.integration
class TestCrossValScoreIntegration:
    """Test integration with sklearn cross_val_score."""

    def test_cross_val_score_with_wrapper(self):
        """Test cross_val_score with wrapped estimator."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)

        scores = cross_val_score(wrapper, X, y, cv=3)

        # Should return 3 scores (one per fold)
        assert len(scores) == 3
        assert all(isinstance(s, int | float) for s in scores)

    def test_cross_val_score_with_pipeline(self):
        """Test cross_val_score with wrapper in Pipeline."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", wrapper)])

        scores = cross_val_score(pipeline, X, y, cv=3)

        assert len(scores) == 3
        assert all(isinstance(s, int | float) for s in scores)


# ============================================================================
# Ensemble Integration Tests
# ============================================================================


@pytest.mark.integration
class TestEnsembleIntegration:
    """Test integration with sklearn ensemble methods."""

    def test_stacking_regressor_with_wrappers(self):
        """Test StackingRegressor with wrapped base estimators."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        # Create wrapped estimators
        wrapper1 = RegressorWrapper(regressor=SimpleRegressorClass, alpha=0.5)
        wrapper2 = RegressorWrapper(regressor=SimpleRegressorClass, alpha=2.0)

        # Use sklearn's Ridge as final estimator
        stacking = StackingRegressor(estimators=[("w1", wrapper1), ("w2", wrapper2)], final_estimator=Ridge())

        stacking.fit(X, y)
        predictions = stacking.predict(X)

        assert predictions.shape == (100,)

    def test_voting_classifier_with_wrappers(self):
        """Test VotingClassifier with wrapped estimators."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        # Create wrapped estimators
        wrapper1 = ClassifierWrapper(classifier=SimpleClassifierClass, threshold=0.3)
        wrapper2 = ClassifierWrapper(classifier=SimpleClassifierClass, threshold=0.7)

        # Use voting classifier
        voting = VotingClassifier(estimators=[("w1", wrapper1), ("w2", wrapper2)], voting="hard")

        voting.fit(X, y)
        predictions = voting.predict(X)

        assert predictions.shape == (100,)

    def test_stacking_with_pipeline(self):
        """Test StackingRegressor with wrapped estimators in pipelines."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        # Create pipelines with wrappers
        pipe1 = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RegressorWrapper(regressor=SimpleRegressorClass, alpha=0.5)),
        ])
        pipe2 = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RegressorWrapper(regressor=SimpleRegressorClass, alpha=2.0)),
        ])

        stacking = StackingRegressor(estimators=[("p1", pipe1), ("p2", pipe2)], final_estimator=Ridge())

        stacking.fit(X, y)
        predictions = stacking.predict(X)

        assert predictions.shape == (100,)


# ============================================================================
# Real-World Scenario Tests
# ============================================================================


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_full_ml_pipeline(self):
        """Test complete ML pipeline: preprocessing + GridSearchCV + evaluation."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        # Split data
        X_train, X_test = X[:80], X[80:]
        y_train, _y_test = y[:80], y[80:]

        # Create pipeline
        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", wrapper)])

        # Grid search
        param_grid = {"regressor__alpha": [0.1, 0.5, 1.0, 2.0]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=3)

        # Fit on training data
        grid_search.fit(X_train, y_train)

        # Predict on test data
        predictions = grid_search.predict(X_test)

        assert predictions.shape == (20,)
        assert hasattr(grid_search, "best_params_")

    def test_nested_cross_validation(self):
        """Test nested cross-validation scenario."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        wrapper = RegressorWrapper(regressor=SimpleRegressorClass, alpha=1.0)

        param_grid = {"alpha": [0.5, 1.0, 2.0]}
        grid_search = GridSearchCV(wrapper, param_grid, cv=3)

        # Outer cross-validation
        scores = cross_val_score(grid_search, X, y, cv=3)

        assert len(scores) == 3
        assert all(isinstance(s, int | float) for s in scores)
