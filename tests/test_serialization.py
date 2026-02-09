"""Serialization tests for BaseClassWrapper.

Tests:
- Pickle serialization/deserialization
- Joblib persistence
- Fitted vs unfitted wrapper serialization
- Nested wrapper serialization
- Parameter preservation after serialization
"""

import pickle
import tempfile
from pathlib import Path

import joblib
import numpy as np

from sklearn_wrap.base import BaseClassWrapper, _fit_context

from .conftest import (
    BaseTestClass,
    ClassWithInner,
    ClassWithNested,
    ClassWithOptional,
    SimpleEstimator,
    SimpleWrapper,
)

# ============================================================================
# Test Wrapper with Fit Method
# ============================================================================


class SerializableWrapper(BaseClassWrapper):
    """Wrapper that can be fitted for serialization testing."""

    _estimator_name = "serializable"
    _estimator_base_class = BaseTestClass

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # Store some fitted attributes
        self.n_features_in_ = len(X[0]) if len(X) > 0 else 0
        return self

    def predict(self, X):
        if hasattr(self.instance_, "predict"):
            return self.instance_.predict(X)
        # Simple prediction for testing
        return np.zeros(len(X))


# ============================================================================
# Pickle Tests - Unfitted Wrapper
# ============================================================================


class TestPickleUnfitted:
    """Test pickling of unfitted wrappers."""

    def test_pickle_unfitted_wrapper(self):
        """Test pickling and unpickling an unfitted wrapper."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5, optional_param=20)

        # Pickle
        pickled = pickle.dumps(wrapper)

        # Unpickle
        unpickled = pickle.loads(pickled)

        # Verify basic attributes
        assert unpickled.estimator_class == SimpleEstimator
        assert unpickled.params["required_param"] == 5
        assert unpickled.params["optional_param"] == 20

    def test_pickle_preserves_parameters(self):
        """Test that pickling preserves all parameters."""
        wrapper = SimpleWrapper(
            simple=SimpleEstimator,
            required_param=42,
            optional_param=100,
            another_optional="custom",
        )

        unpickled = pickle.loads(pickle.dumps(wrapper))

        # All params should match
        assert unpickled.get_params() == wrapper.get_params()
        assert unpickled.params == wrapper.params

    def test_pickle_unfitted_can_be_instantiated(self):
        """Test that unpickled unfitted wrapper can be instantiated."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        unpickled = pickle.loads(pickle.dumps(wrapper))

        # Should be able to instantiate
        unpickled.instantiate()
        assert hasattr(unpickled, "instance_")
        assert isinstance(unpickled.instance_, SimpleEstimator)


# ============================================================================
# Pickle Tests - Fitted Wrapper
# ============================================================================


class TestPickleFitted:
    """Test pickling of fitted wrappers."""

    def test_pickle_fitted_wrapper(self):
        """Test pickling and unpickling a fitted wrapper."""
        wrapper = SerializableWrapper(serializable=SimpleEstimator, required_param=5)

        # Fit the wrapper
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        wrapper.fit(X, y)

        assert wrapper.__sklearn_is_fitted__() is True

        # Pickle
        pickled = pickle.dumps(wrapper)

        # Unpickle
        unpickled = pickle.loads(pickled)

        # Verify it's still fitted
        assert unpickled.__sklearn_is_fitted__() is True
        assert hasattr(unpickled, "instance_")
        assert hasattr(unpickled, "n_features_in_")
        assert unpickled.n_features_in_ == 2

    def test_pickle_fitted_can_predict(self):
        """Test that unpickled fitted wrapper can predict."""
        wrapper = SerializableWrapper(serializable=SimpleEstimator, required_param=5)

        # Fit
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        wrapper.fit(X_train, y_train)

        # Pickle and unpickle
        unpickled = pickle.loads(pickle.dumps(wrapper))

        # Should be able to predict
        X_test = np.array([[2, 3], [4, 5]])
        predictions = unpickled.predict(X_test)

        assert predictions.shape == (2,)

    def test_pickle_fitted_preserves_instance(self):
        """Test that pickling preserves the fitted instance."""
        wrapper = SerializableWrapper(serializable=SimpleEstimator, required_param=5)

        X = np.array([[1, 2], [3, 4]])
        wrapper.fit(X)

        # Store instance attribute before pickling
        wrapper.instance_.custom_attr = "test_value"

        unpickled = pickle.loads(pickle.dumps(wrapper))

        # Instance should be preserved
        assert hasattr(unpickled, "instance_")
        assert hasattr(unpickled.instance_, "custom_attr")
        assert unpickled.instance_.custom_attr == "test_value"


# ============================================================================
# Pickle Tests - Nested Wrappers
# ============================================================================


class TestPickleNested:
    """Test pickling of nested wrapper structures."""

    def test_pickle_nested_wrappers(self):
        """Test pickling wrapper with nested wrapper parameter."""
        inner = SimpleWrapper(simple=SimpleEstimator, required_param=10, optional_param=20)
        outer = SimpleWrapper(simple=ClassWithNested, estimator=inner, value=15)

        # Pickle and unpickle
        unpickled = pickle.loads(pickle.dumps(outer))

        # Verify nested structure is preserved
        assert unpickled.params["value"] == 15
        assert isinstance(unpickled.params["estimator"], SimpleWrapper)
        assert unpickled.params["estimator"].params["required_param"] == 10
        assert unpickled.params["estimator"].params["optional_param"] == 20

    def test_pickle_deep_nested_wrappers(self):
        """Test pickling deeply nested wrapper structure (3 levels)."""
        level1 = SimpleWrapper(simple=SimpleEstimator, required_param=1)
        level2 = SimpleWrapper(simple=ClassWithInner, inner=level1)
        level3 = SimpleWrapper(simple=ClassWithInner, inner=level2)

        # Pickle and unpickle
        unpickled = pickle.loads(pickle.dumps(level3))

        # Verify all levels are preserved
        assert isinstance(unpickled.params["inner"], SimpleWrapper)
        assert isinstance(unpickled.params["inner"].params["inner"], SimpleWrapper)
        assert unpickled.params["inner"].params["inner"].params["required_param"] == 1


# ============================================================================
# Joblib Tests
# ============================================================================


class TestJoblibPersistence:
    """Test joblib save/load functionality."""

    def test_joblib_save_load_unfitted(self):
        """Test joblib save and load with unfitted wrapper."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5, optional_param=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wrapper.joblib"

            # Save
            joblib.dump(wrapper, filepath)

            # Load
            loaded = joblib.load(filepath)

            # Verify
            assert loaded.estimator_class == SimpleEstimator
            assert loaded.params["required_param"] == 5
            assert loaded.params["optional_param"] == 20

    def test_joblib_save_load_fitted(self):
        """Test joblib save and load with fitted wrapper."""
        wrapper = SerializableWrapper(serializable=SimpleEstimator, required_param=5)

        # Fit
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        wrapper.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wrapper.joblib"

            # Save
            joblib.dump(wrapper, filepath)

            # Load
            loaded = joblib.load(filepath)

            # Verify fitted state
            assert loaded.__sklearn_is_fitted__() is True
            assert hasattr(loaded, "instance_")
            assert loaded.n_features_in_ == 2

            # Verify can predict
            X_test = np.array([[2, 3]])
            predictions = loaded.predict(X_test)
            assert predictions.shape == (1,)

    def test_joblib_with_nested_wrappers(self):
        """Test joblib persistence with nested wrappers."""
        inner = SimpleWrapper(simple=SimpleEstimator, required_param=10)
        outer = SimpleWrapper(simple=ClassWithNested, estimator=inner, value=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested_wrapper.joblib"

            # Save
            joblib.dump(outer, filepath)

            # Load
            loaded = joblib.load(filepath)

            # Verify nested structure
            assert loaded.params["value"] == 15
            assert isinstance(loaded.params["estimator"], SimpleWrapper)
            assert loaded.params["estimator"].params["required_param"] == 10

    def test_joblib_compression(self):
        """Test joblib save with compression."""
        wrapper = SerializableWrapper(serializable=SimpleEstimator, required_param=5)

        # Fit to create more data
        X = np.array([[1, 2], [3, 4], [5, 6]] * 100)
        y = np.array([0, 1, 0] * 100)
        wrapper.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "compressed_wrapper.joblib"

            # Save with compression
            joblib.dump(wrapper, filepath, compress=3)

            # Load
            loaded = joblib.load(filepath)

            # Verify
            assert loaded.__sklearn_is_fitted__() is True
            assert loaded.n_features_in_ == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_pickle_wrapper_with_none_params(self):
        """Test pickling wrapper with None parameter values."""
        wrapper = SimpleWrapper(simple=ClassWithOptional, param1=None, param2="value")

        unpickled = pickle.loads(pickle.dumps(wrapper))

        assert unpickled.params["param1"] is None
        assert unpickled.params["param2"] == "value"

    def test_pickle_wrapper_after_set_params(self):
        """Test pickling after modifying parameters with set_params."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        # Modify params
        wrapper.set_params(optional_param=50, another_optional="modified")

        # Pickle
        unpickled = pickle.loads(pickle.dumps(wrapper))

        # Verify modified params are preserved
        assert unpickled.params["optional_param"] == 50
        assert unpickled.params["another_optional"] == "modified"

    def test_roundtrip_multiple_times(self):
        """Test multiple pickle/unpickle cycles."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        # First cycle
        unpickled1 = pickle.loads(pickle.dumps(wrapper))
        assert unpickled1.params["required_param"] == 5

        # Second cycle
        unpickled2 = pickle.loads(pickle.dumps(unpickled1))
        assert unpickled2.params["required_param"] == 5

        # Third cycle
        unpickled3 = pickle.loads(pickle.dumps(unpickled2))
        assert unpickled3.params["required_param"] == 5

        # All should have same params
        assert unpickled3.get_params() == wrapper.get_params()
