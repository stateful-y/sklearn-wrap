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


class TestBasicFitContext:
    """Tests for basic _fit_context decorator functionality."""

    def test_decorator_basic(self):
        """Test the _fit_context decorator functionality."""

        class FittableWrapper(SimpleWrapper):
            def __init__(self, simple, **params):
                super().__init__(simple=simple, **params)
                self.fit_called = False

            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                self.fit_called = True
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]
        y = [0, 1]

        wrapper.fit(X, y)

        assert wrapper.fit_called
        assert hasattr(wrapper, "instance_")
        assert isinstance(wrapper.instance_, SimpleEstimator)

    def test_sets_fitted_flag(self):
        """Test that _fit_context decorator sets fitted flag after successful fit."""

        class FittableWrapper(SimpleWrapper):
            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        assert wrapper.__sklearn_is_fitted__() is False

        wrapper.fit(X)
        assert wrapper.__sklearn_is_fitted__() is True


class TestPartialFit:
    """Tests for partial_fit behavior with _fit_context."""

    def test_skips_reinstantiation_when_fitted(self):
        """Test _fit_context decorator with partial_fit when already fitted."""

        class PartialFittableWrapper(SimpleWrapper):
            def __init__(self, simple, **params):
                super().__init__(simple=simple, **params)
                self.partial_fit_count = 0
                self.instantiate_count = 0

            def instantiate(self):
                self.instantiate_count += 1
                return super().instantiate()

            @_fit_context(prefer_skip_nested_validation=True)
            def partial_fit(self, X, y=None):
                self.partial_fit_count += 1
                return self

        wrapper = PartialFittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]
        y = [0, 1]

        wrapper.partial_fit(X, y)
        assert wrapper.partial_fit_count == 1
        assert wrapper.instantiate_count == 1

        wrapper.is_fitted_ = True

        wrapper.partial_fit(X, y)
        assert wrapper.partial_fit_count == 2
        assert wrapper.instantiate_count == 1

    def test_reinstantiates_when_not_fitted(self):
        """Test that partial_fit reinstantiates if estimator is not fitted."""

        class PartialFittableWrapper(SimpleWrapper):
            def __init__(self, simple, **params):
                super().__init__(simple=simple, **params)
                self.instantiate_count = 0

            def instantiate(self):
                self.instantiate_count += 1
                return super().instantiate()

            @_fit_context(prefer_skip_nested_validation=True)
            def partial_fit(self, X, y=None):
                return self

        wrapper = PartialFittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        wrapper.partial_fit(X)
        assert wrapper.instantiate_count == 1

        wrapper.partial_fit(X)
        assert wrapper.instantiate_count == 2


class TestNonEstimatorObjects:
    """Tests for _fit_context on objects without instantiate method."""

    def test_without_instantiate(self):
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

        estimator.fit(X, y)
        assert estimator.fit_called


class TestConfigContext:
    """Tests for _fit_context interaction with sklearn config_context."""

    def test_with_skip_validation_config(self):
        """Test _fit_context decorator with global skip_parameter_validation config."""

        class FittableWrapper(SimpleWrapper):
            def __init__(self, simple, **params):
                super().__init__(simple=simple, **params)
                self.fit_called = False
                self.validate_params_called = False

            def _validate_params(self):
                self.validate_params_called = True
                super()._validate_params()

            @_fit_context(prefer_skip_nested_validation=False)
            def fit(self, X, y=None):
                self.fit_called = True
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]
        y = [0, 1]

        with config_context(skip_parameter_validation=True):
            wrapper.fit(X, y)
            assert wrapper.fit_called
            assert wrapper.validate_params_called

    def test_prefer_skip_nested_validation_true(self):
        """Test _fit_context with prefer_skip_nested_validation=True."""

        class FittableWrapper(SimpleWrapper):
            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        wrapper.fit(X)
        assert hasattr(wrapper, "instance_")

    def test_prefer_skip_nested_validation_false(self):
        """Test _fit_context with prefer_skip_nested_validation=False."""

        class FittableWrapper(SimpleWrapper):
            @_fit_context(prefer_skip_nested_validation=False)
            def fit(self, X, y=None):
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        wrapper.fit(X)
        assert hasattr(wrapper, "instance_")


class TestFitContextEdgeCases:
    """Tests for _fit_context edge cases."""

    def test_exception_does_not_set_fitted(self):
        """Test that _fit_context doesn't set fitted flag if fit raises exception."""

        class FittableWrapper(SimpleWrapper):
            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                raise ValueError("Intentional error")

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        with pytest.raises(ValueError, match="Intentional error"):
            wrapper.fit(X)

        assert wrapper.__sklearn_is_fitted__() is False

    def test_multiple_fits(self):
        """Test calling fit multiple times with _fit_context."""

        class FittableWrapper(SimpleWrapper):
            def __init__(self, simple, **params):
                super().__init__(simple=simple, **params)
                self.fit_count = 0

            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                self.fit_count += 1
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        X = [[1, 2], [3, 4]]

        wrapper.fit(X)
        assert wrapper.fit_count == 1
        assert wrapper.__sklearn_is_fitted__() is True

        wrapper.fit(X)
        assert wrapper.fit_count == 2
        assert wrapper.__sklearn_is_fitted__() is True

    def test_validates_estimator_class(self):
        """Test that _fit_context triggers validation via instantiate."""

        class FittableWrapper(BaseClassWrapper):
            _estimator_name = "simple"
            _estimator_base_class = BaseTestClass

            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                return self

        wrapper = FittableWrapper(simple=SimpleEstimator, required_param=5)
        wrapper.estimator_class = NotBaseClass

        X = [[1, 2], [3, 4]]

        with pytest.raises(ValueError, match="should be derived from"):
            wrapper.fit(X)

    def test_fit_with_inplace_modification(self):
        """Test _fit_context when fit method modifies X in place."""

        class InplaceEstimator(BaseTestClass):
            def __init__(self, scale=1):
                self.scale = scale

            def fit(self, X, y=None):
                for row in X:
                    for i in range(len(row)):
                        row[i] *= self.scale
                return self

        class InplaceWrapper(BaseClassWrapper):
            _estimator_name = "inplace"
            _estimator_base_class = BaseTestClass

            @_fit_context(prefer_skip_nested_validation=True)
            def fit(self, X, y=None):
                self.instance_.fit(X, y)
                return self

        wrapper = InplaceWrapper(inplace=InplaceEstimator, scale=2)
        X = [[1, 2], [3, 4]]
        wrapper.fit(X)

        assert wrapper.__sklearn_is_fitted__() is True
        assert X == [[2, 4], [6, 8]]
