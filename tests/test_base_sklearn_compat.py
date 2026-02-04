"""Sklearn compatibility tests for BaseClassWrapper.

Tests:
- HTML representation (_repr_html_)
- Cloning (sklearn.base.clone)
- Fitted state detection (__sklearn_is_fitted__)
- Integration with sklearn utilities
"""

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from sklearn_wrap.base import BaseClassWrapper

from .conftest import BaseTestClass, SimpleEstimator, SimpleWrapper

# ============================================================================
# HTML Representation Tests
# ============================================================================


class TestHTMLRepresentation:
    """Tests for HTML representation in Jupyter notebooks."""

    def test_html_repr_method_exists(self):
        """Test that _repr_html_ method exists (inherited from BaseEstimator)."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        assert hasattr(wrapper, "_repr_html_")
        assert callable(wrapper._repr_html_)

    def test_html_repr_renders(self):
        """Test that _repr_html_ returns a string with HTML content."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, optional_param=20)
        html = wrapper._repr_html_()

        assert isinstance(html, str)
        assert len(html) > 0
        # Basic check that it contains some HTML-like content
        assert "<" in html and ">" in html

    def test_html_repr_with_nested_wrapper(self):
        """Test HTML representation with nested BaseClassWrapper instances."""
        # Create nested wrapper structure
        inner_wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=10)

        class ClassWithNestedParam:
            def __init__(self, nested_estimator, value=5):
                self.nested_estimator = nested_estimator
                self.value = value

        class OuterWrapper(BaseClassWrapper):
            _estimator_name = "outer"
            _estimator_base_class = object

        outer_wrapper = OuterWrapper(estimator_class=ClassWithNestedParam, nested_estimator=inner_wrapper, value=15)

        # Should not raise and should produce HTML
        html = outer_wrapper._repr_html_()
        assert isinstance(html, str)
        assert len(html) > 0


# ============================================================================
# Cloning Tests
# ============================================================================


class TestCloning:
    """Tests for sklearn.base.clone() compatibility."""

    def test_clone_creates_independent_copy(self):
        """Test that clone() creates an independent copy of the wrapper."""
        original = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, optional_param=20)
        cloned = clone(original)

        # Should be different objects
        assert cloned is not original
        assert cloned.params is not original.params

    def test_clone_preserves_estimator_class(self):
        """Test that clone() preserves the estimator_class."""
        original = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        cloned = clone(original)

        assert cloned.estimator_class is original.estimator_class
        assert cloned.estimator_class == SimpleEstimator

    def test_clone_preserves_parameters(self):
        """Test that clone() preserves all parameters."""
        original = SimpleWrapper(
            estimator_class=SimpleEstimator,
            required_param=42,
            optional_param=100,
            another_optional="custom",
        )
        cloned = clone(original)

        # Check parameters match
        assert cloned.get_params() == original.get_params()
        assert cloned.params["required_param"] == 42
        assert cloned.params["optional_param"] == 100
        assert cloned.params["another_optional"] == "custom"

    def test_clone_resets_fitted_state(self):
        """Test that clone() removes instance_ attribute (unfits the estimator)."""
        # Create and instantiate wrapper
        original = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        original.instantiate()

        # Verify original has instance_
        assert hasattr(original, "instance_")

        # Clone should not have instance_
        cloned = clone(original)
        assert not hasattr(cloned, "instance_")

    def test_clone_resets_fitted_flag(self):
        """Test that clone() resets the _fitted flag."""
        # Create wrapper and mark as fitted
        original = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        original._fitted = True

        assert original.__sklearn_is_fitted__() is True

        # Clone should not be fitted
        cloned = clone(original)
        assert cloned.__sklearn_is_fitted__() is False
        assert not hasattr(cloned, "_fitted") or cloned._fitted is False

    def test_clone_with_nested_wrappers(self):
        """Test cloning with nested BaseClassWrapper instances."""
        # Create nested structure
        inner = SimpleWrapper(estimator_class=SimpleEstimator, required_param=10)

        class ClassWithNestedParam:
            def __init__(self, nested, value=5):
                self.nested = nested
                self.value = value

        class OuterWrapper(BaseClassWrapper):
            _estimator_name = "outer"
            _estimator_base_class = object

        original = OuterWrapper(estimator_class=ClassWithNestedParam, nested=inner, value=15)

        cloned = clone(original)

        # Outer wrapper should be cloned
        assert cloned is not original

        # Inner wrapper should also be cloned
        assert cloned.params["nested"] is not original.params["nested"]
        assert isinstance(cloned.params["nested"], SimpleWrapper)

        # But should have same parameters
        assert cloned.params["nested"].get_params() == inner.get_params()


# ============================================================================
# Fitted State Detection Tests
# ============================================================================


class TestFittedStateDetection:
    """Tests for fitted state detection with check_is_fitted()."""

    def test_sklearn_is_fitted_method_exists(self):
        """Test that __sklearn_is_fitted__() method exists."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        assert hasattr(wrapper, "__sklearn_is_fitted__")
        assert callable(wrapper.__sklearn_is_fitted__)

    def test_unfitted_wrapper_not_fitted(self):
        """Test that newly created wrapper is not fitted."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        assert wrapper.__sklearn_is_fitted__() is False

    def test_instantiated_wrapper_not_fitted(self):
        """Test that instantiate() alone does not mark wrapper as fitted."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        wrapper.instantiate()

        # Should have instance_ but not be fitted
        assert hasattr(wrapper, "instance_")
        assert wrapper.__sklearn_is_fitted__() is False

    def test_check_is_fitted_raises_before_fit(self):
        """Test that check_is_fitted() raises NotFittedError before fitting."""

        # Create a wrapper with a fit method
        class WrapperWithFit(BaseClassWrapper):
            _estimator_name = "simple"
            _estimator_base_class = BaseTestClass

            def fit(self, X, y=None):
                self.instantiate()
                self._fitted = True
                return self

        wrapper = WrapperWithFit(estimator_class=SimpleEstimator, required_param=5)

        with pytest.raises(NotFittedError):
            check_is_fitted(wrapper)

    def test_check_is_fitted_raises_after_instantiate_only(self):
        """Test that check_is_fitted() raises even after instantiate()."""

        class WrapperWithFit(BaseClassWrapper):
            _estimator_name = "simple"
            _estimator_base_class = BaseTestClass

            def fit(self, X, y=None):
                self.instantiate()
                self._fitted = True
                return self

        wrapper = WrapperWithFit(estimator_class=SimpleEstimator, required_param=5)
        wrapper.instantiate()

        # Still should raise because _fitted flag is not set
        with pytest.raises(NotFittedError):
            check_is_fitted(wrapper)

    def test_fitted_after_setting_flag(self):
        """Test that wrapper is fitted after setting _fitted flag."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
        wrapper.instantiate()
        wrapper._fitted = True

        assert wrapper.__sklearn_is_fitted__() is True

    def test_check_is_fitted_passes_after_fitting(self):
        """Test that check_is_fitted() passes after setting _fitted flag."""

        class WrapperWithFit(BaseClassWrapper):
            _estimator_name = "simple"
            _estimator_base_class = BaseTestClass

            def fit(self, X, y=None):
                self.instantiate()
                self._fitted = True
                return self

        wrapper = WrapperWithFit(estimator_class=SimpleEstimator, required_param=5)
        wrapper.instantiate()
        wrapper._fitted = True

        # Should not raise
        check_is_fitted(wrapper)

    def test_fitted_flag_persists(self):
        """Test that _fitted flag persists across calls."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

        assert wrapper.__sklearn_is_fitted__() is False

        wrapper._fitted = True
        assert wrapper.__sklearn_is_fitted__() is True
        assert wrapper.__sklearn_is_fitted__() is True  # Still True on second call

        wrapper._fitted = False
        assert wrapper.__sklearn_is_fitted__() is False

    def test_instantiate_resets_fitted_flag(self):
        """Test that instantiate() resets the _fitted flag."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

        # Fit and mark as fitted
        wrapper.instantiate()
        wrapper._fitted = True
        assert wrapper.__sklearn_is_fitted__() is True
        assert hasattr(wrapper, "instance_")

        # Call instantiate() again - should reset fitted flag
        wrapper.instantiate()
        assert wrapper.__sklearn_is_fitted__() is False
        assert hasattr(wrapper, "instance_")  # instance_ still exists but is new
