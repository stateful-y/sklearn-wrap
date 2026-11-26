"""Integration tests: FunctionWrapper subclass with mixins in Pipeline, GridSearchCV."""

import numpy as np
import pytest
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn_wrap.base import FunctionWrapper, _fit_context

from .conftest import FitPredictFunctionWrapper, predict_fn


class RegressorFunctionWrapper(FunctionWrapper, RegressorMixin):
    """Full estimator built from FunctionWrapper with _fit_context."""

    _callable_name = "fn"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.callable_fn(X, **self._params)


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = X.sum(axis=1) * 2.0 + 1.0
    return X, y


class TestFitContext:
    """Tests for _fit_context integration with FunctionWrapper."""

    def test_fit_context_calls_instantiate(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn, scale=2.0, offset=1.0)
        wrapper.fit(X, y)
        assert wrapper.__sklearn_is_fitted__()

    def test_fit_context_sets_fitted(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn)
        assert not wrapper.__sklearn_is_fitted__()
        wrapper.fit(X, y)
        assert wrapper.__sklearn_is_fitted__()

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn, scale=2.0, offset=1.0)
        wrapper.fit(X, y)
        predictions = wrapper.predict(X)
        expected = X.sum(axis=1) * 2.0 + 1.0
        np.testing.assert_array_almost_equal(predictions, expected)


class TestPipeline:
    """Tests for FunctionWrapper in sklearn Pipeline."""

    def test_pipeline_fit_predict(self, sample_data):
        X, y = sample_data
        pipe = Pipeline([("regressor", RegressorFunctionWrapper(fn=predict_fn, scale=1.0))])
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (50,)

    def test_pipeline_set_params(self, sample_data):
        X, y = sample_data
        pipe = Pipeline([("regressor", RegressorFunctionWrapper(fn=predict_fn, scale=1.0))])
        pipe.set_params(regressor__scale=3.0)
        assert pipe.named_steps["regressor"]._params["scale"] == 3.0

    def test_pipeline_get_params(self):
        pipe = Pipeline([("regressor", RegressorFunctionWrapper(fn=predict_fn, scale=2.0))])
        params = pipe.get_params()
        assert params["regressor__scale"] == 2.0
        assert params["regressor__offset"] == 0.0


class TestGridSearchCV:
    """Tests for FunctionWrapper with GridSearchCV."""

    def test_grid_search(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn, scale=1.0, offset=0.0)
        grid = GridSearchCV(
            wrapper,
            param_grid={"scale": [0.5, 1.0, 2.0], "offset": [0.0, 1.0]},
            cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X, y)
        assert "scale" in grid.best_params_
        assert "offset" in grid.best_params_

    def test_cross_val_score(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn, scale=2.0, offset=1.0)
        scores = cross_val_score(wrapper, X, y, cv=3, scoring="neg_mean_squared_error")
        assert len(scores) == 3


class TestCloneEstimator:
    """Tests for clone of estimator-style FunctionWrapper."""

    def test_clone_regressor_wrapper(self, sample_data):
        X, y = sample_data
        wrapper = RegressorFunctionWrapper(fn=predict_fn, scale=2.0, offset=1.0)
        wrapper.fit(X, y)
        cloned = clone(wrapper)
        assert not cloned.__sklearn_is_fitted__()
        assert cloned.callable_fn is predict_fn
        assert cloned._params["scale"] == 2.0

    def test_clone_in_pipeline(self, sample_data):
        X, y = sample_data
        pipe = Pipeline([("regressor", RegressorFunctionWrapper(fn=predict_fn, scale=1.5))])
        cloned_pipe = clone(pipe)
        cloned_pipe.fit(X, y)
        predictions = cloned_pipe.predict(X)
        assert predictions.shape == (50,)


class TestFitPredictWrapper:
    """Tests for the simpler FitPredictFunctionWrapper fixture (no _fit_context)."""

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        wrapper = FitPredictFunctionWrapper(fn=predict_fn, scale=2.0, offset=0.5)
        wrapper.fit(X, y)
        assert wrapper.__sklearn_is_fitted__()
        predictions = wrapper.predict(X)
        expected = X.sum(axis=1) * 2.0 + 0.5
        np.testing.assert_array_almost_equal(predictions, expected)
