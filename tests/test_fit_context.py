"""Tests for _fit_context decorator.

Tests:
- Basic decorator functionality
- partial_fit behavior (no re-instantiation when fitted)
- Objects without instantiate method
- Global skip_parameter_validation config
- Integration with fit methods
"""

import pytest
from sklearn._config import config_context

from sklearn_wrap.base import BaseClassWrapper, _fit_context

from .conftest import BaseTestClass, NotBaseClass, SimpleEstimator, SimpleWrapper

# ============================================================================
# Basic _fit_context Decorator Tests
# ============================================================================


def test_fit_context_decorator_basic():
    """Test the _fit_context decorator functionality."""

    class FittableWrapper(SimpleWrapper):
        def __init__(self, estimator_class, **params):
            super().__init__(estimator_class, **params)
            self.fit_called = False

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            self.fit_called = True
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Call fit - should trigger instantiate via decorator
    wrapper.fit(X, y)

    assert wrapper.fit_called
    assert hasattr(wrapper, "instance_")
    assert isinstance(wrapper.instance_, SimpleEstimator)


def test_fit_context_sets_fitted_flag():
    """Test that _fit_context decorator sets _fitted flag after successful fit."""

    class FittableWrapper(SimpleWrapper):
        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # Before fit
    assert wrapper.__sklearn_is_fitted__() is False

    # After fit
    wrapper.fit(X)
    assert wrapper.__sklearn_is_fitted__() is True


# ============================================================================
# partial_fit Tests
# ============================================================================


def test_fit_context_decorator_partial_fit():
    """Test _fit_context decorator with partial_fit when already fitted."""

    class PartialFittableWrapper(SimpleWrapper):
        def __init__(self, estimator_class, **params):
            super().__init__(estimator_class, **params)
            self.partial_fit_count = 0
            self.instantiate_count = 0

        def instantiate(self):
            self.instantiate_count += 1
            return super().instantiate()

        @_fit_context(prefer_skip_nested_validation=True)
        def partial_fit(self, X, y=None):
            self.partial_fit_count += 1
            return self

    wrapper = PartialFittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # First call to partial_fit - should instantiate
    wrapper.partial_fit(X, y)
    assert wrapper.partial_fit_count == 1
    assert wrapper.instantiate_count == 1

    # Mark as fitted
    wrapper.is_fitted_ = True

    # Second call to partial_fit - should NOT instantiate again
    wrapper.partial_fit(X, y)
    assert wrapper.partial_fit_count == 2
    # instantiate_count should still be 1 because it's already fitted
    assert wrapper.instantiate_count == 1


def test_fit_context_partial_fit_reinstantiates_when_not_fitted():
    """Test that partial_fit reinstantiates if estimator is not fitted."""

    class PartialFittableWrapper(SimpleWrapper):
        def __init__(self, estimator_class, **params):
            super().__init__(estimator_class, **params)
            self.instantiate_count = 0

        def instantiate(self):
            self.instantiate_count += 1
            return super().instantiate()

        @_fit_context(prefer_skip_nested_validation=True)
        def partial_fit(self, X, y=None):
            return self

    wrapper = PartialFittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # First call - should instantiate
    wrapper.partial_fit(X)
    assert wrapper.instantiate_count == 1

    # Don't set is_fitted_ - call again
    wrapper.partial_fit(X)
    # Should instantiate again since not marked as fitted
    assert wrapper.instantiate_count == 2


# ============================================================================
# Objects Without instantiate Method
# ============================================================================


def test_fit_context_decorator_without_instantiate():
    """Test _fit_context decorator on object without instantiate method."""

    class SimpleEstimatorClass:
        def __init__(self):
            self.fit_called = False

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            self.fit_called = True
            return self

    estimator = SimpleEstimatorClass()
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Should work fine even without instantiate method
    estimator.fit(X, y)
    assert estimator.fit_called


# ============================================================================
# Config Context Tests
# ============================================================================


def test_fit_context_decorator_with_skip_validation_config():
    """Test _fit_context decorator with global skip_parameter_validation config."""

    class FittableWrapper(SimpleWrapper):
        def __init__(self, estimator_class, **params):
            super().__init__(estimator_class, **params)
            self.fit_called = False
            self.validate_params_called = False

        def _validate_params(self):
            self.validate_params_called = True
            super()._validate_params()

        @_fit_context(prefer_skip_nested_validation=False)
        def fit(self, X, y=None):
            self.fit_called = True
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Test with skip_parameter_validation=True
    with config_context(skip_parameter_validation=True):
        wrapper.fit(X, y)
        assert wrapper.fit_called
        # instantiate() is called which calls _validate_params
        assert wrapper.validate_params_called


def test_fit_context_prefer_skip_nested_validation():
    """Test _fit_context with prefer_skip_nested_validation parameter."""

    class FittableWrapper(SimpleWrapper):
        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # Should work with prefer_skip_nested_validation=True
    wrapper.fit(X)
    assert hasattr(wrapper, "instance_")


def test_fit_context_without_prefer_skip_nested_validation():
    """Test _fit_context with prefer_skip_nested_validation=False."""

    class FittableWrapper(SimpleWrapper):
        @_fit_context(prefer_skip_nested_validation=False)
        def fit(self, X, y=None):
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # Should still work
    wrapper.fit(X)
    assert hasattr(wrapper, "instance_")


# ============================================================================
# Edge Cases
# ============================================================================


def test_fit_context_with_exception():
    """Test that _fit_context doesn't set fitted flag if fit raises exception."""

    class FittableWrapper(SimpleWrapper):
        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            raise ValueError("Intentional error")

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # Fit raises exception
    with pytest.raises(ValueError, match="Intentional error"):
        wrapper.fit(X)

    # Should not be fitted
    assert wrapper.__sklearn_is_fitted__() is False


def test_fit_context_multiple_fits():
    """Test calling fit multiple times with _fit_context."""

    class FittableWrapper(SimpleWrapper):
        def __init__(self, estimator_class, **params):
            super().__init__(estimator_class, **params)
            self.fit_count = 0

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            self.fit_count += 1
            return self

    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)
    X = [[1, 2], [3, 4]]

    # First fit
    wrapper.fit(X)
    assert wrapper.fit_count == 1
    assert wrapper.__sklearn_is_fitted__() is True

    # Second fit - should re-instantiate and fit again
    wrapper.fit(X)
    assert wrapper.fit_count == 2
    # Should still be fitted after second fit
    assert wrapper.__sklearn_is_fitted__() is True


def test_fit_context_validates_estimator_class():
    """Test that _fit_context triggers validation via instantiate."""

    class FittableWrapper(BaseClassWrapper):
        _estimator_name = "simple"
        _estimator_base_class = BaseTestClass

        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            return self

    # Create wrapper with valid class
    wrapper = FittableWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Manually change to invalid class (bypass __init__ validation)
    wrapper.estimator_class = NotBaseClass

    X = [[1, 2], [3, 4]]

    # Fit should trigger validation and raise error
    with pytest.raises(ValueError, match="should be derived from"):
        wrapper.fit(X)
