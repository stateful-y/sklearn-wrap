"""Parameter management tests for BaseClassWrapper.

Tests:
- get_params method (shallow and deep)
- set_params method (simple and nested)
- Parameter validation and constraints
- Double underscore handling for nested params
- Roundtrip compatibility with sklearn
"""

import pytest

from sklearn_wrap.base import BaseClassWrapper

from .conftest import BaseTestClass, NoRequiredParams, SimpleEstimator, SimpleWrapper


class TestGetParams:
    """Tests for the get_params method."""

    def test_returns_dict(self):
        """Test that get_params returns a dictionary."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        params = wrapper.get_params()
        assert isinstance(params, dict)

    def test_includes_estimator_name(self):
        """Test that get_params includes the estimator name key."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        params = wrapper.get_params()
        assert "simple" in params
        assert params["simple"] == SimpleEstimator

    def test_includes_all_params(self):
        """Test that get_params includes all constructor parameters."""
        wrapper = SimpleWrapper(
            simple=SimpleEstimator,
            required_param=5,
            optional_param=20,
            another_optional="test",
        )
        params = wrapper.get_params()

        assert params["required_param"] == 5
        assert params["optional_param"] == 20
        assert params["another_optional"] == "test"

    def test_includes_defaults(self):
        """Test that get_params includes default parameter values."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        params = wrapper.get_params()

        assert params["optional_param"] == 10
        assert params["another_optional"] == "default"

    def test_with_deep_parameter(self):
        """Test that get_params accepts deep parameter."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        params_deep = wrapper.get_params(deep=True)
        params_shallow = wrapper.get_params(deep=False)

        assert "required_param" in params_deep
        assert "required_param" in params_shallow

    def test_deep_with_nested_estimator(self):
        """Test get_params with deep=True and nested estimators."""
        inner_wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=1)

        class ClassWithNestedEstimator(BaseTestClass):
            def __init__(self, nested=None, simple_param=5):
                self.nested = nested
                self.simple_param = simple_param

        wrapper = SimpleWrapper(
            simple=ClassWithNestedEstimator,
            nested=inner_wrapper,
            simple_param=10,
        )

        params = wrapper.get_params(deep=True)
        assert "nested" in params
        assert "simple_param" in params
        assert params["nested"] == inner_wrapper
        assert "nested__required_param" in params
        assert params["nested__required_param"] == 1

    def test_deep_with_non_wrapper_nested(self):
        """Test get_params with deep=True when nested object has get_params but isn't a wrapper."""

        class NonWrapperWithGetParams:
            """A class with get_params but not a BaseClassWrapper."""

            def __init__(self, value=42):
                self.value = value

            def get_params(self, deep=True):
                return {"value": self.value}

        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper.params["custom_obj"] = NonWrapperWithGetParams(value=100)

        params = wrapper.get_params(deep=True)
        assert "custom_obj__value" in params
        assert params["custom_obj__value"] == 100


class TestSetParams:
    """Tests for the set_params method."""

    def test_updates_params(self):
        """Test that set_params updates parameters correctly."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper.set_params(optional_param=30)

        assert wrapper.params["optional_param"] == 30

    def test_returns_self(self):
        """Test that set_params returns self for chaining."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        result = wrapper.set_params(optional_param=30)

        assert result is wrapper

    def test_multiple_params(self):
        """Test setting multiple parameters at once."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper.set_params(optional_param=25, another_optional="new")

        assert wrapper.params["optional_param"] == 25
        assert wrapper.params["another_optional"] == "new"

    def test_empty(self):
        """Test set_params with no arguments."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        original_class = wrapper.estimator_class

        wrapper.set_params()

        assert wrapper.estimator_class == original_class
        assert "required_param" in wrapper.params
        assert "optional_param" in wrapper.params

    def test_then_instantiate(self):
        """Test changing params and then instantiating."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper.set_params(required_param=5, optional_param=77)
        wrapper.instantiate()

        assert wrapper.instance_.required_param == 5
        assert wrapper.instance_.optional_param == 77

    def test_changes_estimator_class_raises(self):
        """Test that set_params raises error when trying to change the estimator class."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="Cannot change estimator class via set_params"):
            wrapper.set_params(simple=NoRequiredParams)

    def test_validates_new_params(self):
        """Test that set_params validates new parameters."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="'invalid_param' is not a valid parameter for class 'SimpleEstimator'"):
            wrapper.set_params(invalid_param=100)

    def test_with_estimator_class_key(self):
        """Test that set_params rejects unknown 'estimator_class' parameter."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="'estimator_class' is not a valid parameter for class 'SimpleEstimator'"):
            wrapper.set_params(estimator_class=NoRequiredParams)

    def test_with_non_type_estimator_value(self):
        """Test that set_params raises error when trying to set estimator name."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="Cannot change estimator class via set_params"):
            wrapper.set_params(simple="not_a_type")

    def test_invalid_param_after_validation(self):
        """Test error when param doesn't exist after validation."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(
            ValueError, match="'totally_invalid_param' is not a valid parameter for class 'SimpleEstimator'"
        ):
            wrapper.set_params(totally_invalid_param=100)

    def test_with_none_value_and_constraints(self):
        """Test set_params with None value when constraints are defined."""

        class TestWrapper(BaseClassWrapper):
            _estimator_name = "test"
            _estimator_base_class = BaseTestClass
            _parameter_constraints = {"optional": [{"wrapper_base_class": BaseTestClass}]}

        class OuterClass(BaseTestClass):
            def __init__(self, optional=None):
                self.optional = optional

        wrapper = TestWrapper(test=OuterClass, optional=None)

        wrapper.set_params(optional=None)
        assert wrapper.params["optional"] is None


class TestNestedParams:
    """Tests for nested parameter setting with __ syntax."""

    def test_basic(self):
        """Test basic nested parameter setting with __ syntax."""
        inner_wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=1, optional_param=10)

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None, other_param=5):
                self.estimator = estimator
                self.other_param = other_param

        outer_wrapper = SimpleWrapper(simple=ClassWithNested, estimator=inner_wrapper, other_param=10)

        outer_wrapper.set_params(estimator__optional_param=100)

        assert outer_wrapper.params["estimator"].params["optional_param"] == 100
        assert outer_wrapper.params["other_param"] == 10

    def test_multiple_levels(self):
        """Test multi-level nested parameter setting."""
        level1 = SimpleWrapper(simple=SimpleEstimator, required_param=1, optional_param=10)

        class ClassWithNested(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        level2 = SimpleWrapper(simple=ClassWithNested, inner=level1)
        level3 = SimpleWrapper(simple=ClassWithNested, inner=level2)

        level3.set_params(inner__inner__optional_param=999)

        assert level3.params["inner"].params["inner"].params["optional_param"] == 999

    def test_nested_and_simple_mixed(self):
        """Test setting both nested and simple parameters together."""
        inner = SimpleWrapper(simple=SimpleEstimator, required_param=1, optional_param=10)

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None, other_param=5):
                self.estimator = estimator
                self.other_param = other_param

        outer = SimpleWrapper(simple=ClassWithNested, estimator=inner, other_param=10)

        outer.set_params(estimator__optional_param=200, other_param=20)

        assert outer.params["estimator"].params["optional_param"] == 200
        assert outer.params["other_param"] == 20

    def test_estimator_object(self):
        """Test set_params with an estimator object that has get_params/set_params."""
        inner_wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=1, optional_param=10)

        class ClassWithEstimatorParam(BaseTestClass):
            def __init__(self, estimator=None, other_param=5):
                self.estimator = estimator
                self.other_param = other_param

        outer_wrapper = SimpleWrapper(simple=ClassWithEstimatorParam, estimator=inner_wrapper, other_param=10)

        outer_wrapper.set_params(other_param=20)
        assert outer_wrapper.params["other_param"] == 20
        assert outer_wrapper.params["estimator"] == inner_wrapper

    def test_without_set_params_method(self):
        """Test error when trying to set nested params on object without set_params."""

        class ClassWithScalar(BaseTestClass):
            def __init__(self, scalar_param=5):
                self.scalar_param = scalar_param

        wrapper = SimpleWrapper(simple=ClassWithScalar, scalar_param=10)

        with pytest.raises(AttributeError, match="does not have a set_params method"):
            wrapper.set_params(scalar_param__something=100)

    def test_invalid_base_key(self):
        """Test error when trying to set nested param on non-existent base param."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="Invalid parameter 'nested' for estimator"):
            wrapper.set_params(nested__optional_param=100)


class TestParameterConstraints:
    """Tests for _parameter_constraints validation."""

    def test_wrapper_base_class(self):
        """Test parameter constraints for nested wrapper validation."""

        class SpecializedWrapper(SimpleWrapper):
            _parameter_constraints = {"estimator": [{"wrapper_base_class": BaseTestClass}]}

        inner = SimpleWrapper(simple=SimpleEstimator, required_param=1)

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None):
                self.estimator = estimator

        outer = SpecializedWrapper(simple=ClassWithNested, estimator=inner)
        outer.set_params(estimator__optional_param=100)

    def test_not_wrapper(self):
        """Test parameter constraints reject non-wrapper values."""

        class SpecializedWrapper(SimpleWrapper):
            _parameter_constraints = {"estimator": [{"wrapper_base_class": BaseTestClass}]}

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None):
                self.estimator = estimator

        with pytest.raises(TypeError, match="must be a BaseClassWrapper instance"):
            SpecializedWrapper(simple=ClassWithNested, estimator="not a wrapper")

    def test_wrong_base_class(self):
        """Test parameter constraints reject wrapper with wrong base class."""

        class OtherBaseClass:
            pass

        class OtherDummyClass(OtherBaseClass):
            def __init__(self, param=1):
                self.param = param

        class OtherWrapper(BaseClassWrapper):
            _estimator_name = "other"
            _estimator_base_class = OtherBaseClass

        class SpecializedWrapper(SimpleWrapper):
            _parameter_constraints = {"estimator": [{"wrapper_base_class": BaseTestClass}]}

        wrong_inner = OtherWrapper(other=OtherDummyClass, param=1)

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None):
                self.estimator = estimator

        with pytest.raises(ValueError, match="must wrap an estimator class derived from"):
            SpecializedWrapper(simple=ClassWithNested, estimator=wrong_inner)

    def test_empty_constraints(self):
        """Test that wrapper works with no parameter constraints."""

        class WrapperNoConstraints(SimpleWrapper):
            _parameter_constraints = {}

        wrapper = WrapperNoConstraints(simple=SimpleEstimator, required_param=5)
        assert wrapper.params["required_param"] == 5

    def test_non_dict_constraint_skipped(self):
        """Test that non-dict constraints are skipped in _validate_nested_wrapper_param."""

        class ClassWithParam(BaseTestClass):
            def __init__(self, param=None):
                self.param = param

        class TestWrapper(BaseClassWrapper):
            _estimator_name = "test"
            _estimator_base_class = BaseTestClass
            _parameter_constraints = {"param": ["not_a_dict"]}

        wrapper = TestWrapper(test=ClassWithParam, param="value")
        assert wrapper.params["param"] == "value"

    def test_non_matching_constraint(self):
        """Test parameter constraint that doesn't match actual parameter."""

        class WrapperWithNonMatchingConstraint(SimpleWrapper):
            _parameter_constraints = {"nonexistent_param": [{"wrapper_base_class": BaseTestClass}]}

        wrapper = WrapperWithNonMatchingConstraint(simple=SimpleEstimator, required_param=5)
        assert wrapper.params["required_param"] == 5

    def test_non_wrapper_class_raises(self):
        """Test that passing a class (not instance) raises TypeError when constraint expects wrapper."""

        class InnerClass(BaseTestClass):
            def __init__(self, value=1):
                self.value = value

        class OuterClass(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        class StrictWrapper(BaseClassWrapper):
            _estimator_name = "strict"
            _estimator_base_class = BaseTestClass
            _parameter_constraints = {"inner": [{"wrapper_base_class": BaseTestClass}]}

        with pytest.raises(TypeError, match="must be a BaseClassWrapper instance"):
            StrictWrapper(strict=OuterClass, inner=InnerClass)


class TestDoubleUnderscoreValidation:
    """Tests for double underscore validation in parameter names."""

    def test_parameter_names_cannot_contain_double_underscore(self):
        """Test that parameter names cannot contain __ which is reserved for nested syntax."""

        class ClassWithInvalidParam(BaseTestClass):
            def __init__(self, invalid__param=5):
                self.invalid__param = invalid__param

        with pytest.raises(ValueError, match="cannot contain '__'"):
            SimpleWrapper(simple=ClassWithInvalidParam, invalid__param=10)

    def test_validate_directly(self):
        """Test direct validation of params dict with __ in name."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        with pytest.raises(ValueError, match="cannot contain '__'"):
            wrapper._validate_estimator_params({"some__invalid__param": 100})


class TestRoundtripCompatibility:
    """Tests for get_params/set_params roundtrip compatibility."""

    def test_roundtrip(self):
        """Test that get_params/set_params roundtrip works correctly."""
        wrapper = SimpleWrapper(
            simple=SimpleEstimator,
            required_param=5,
            optional_param=20,
            another_optional="test",
        )

        params = wrapper.get_params()
        params.pop("simple", None)
        wrapper.set_params(**params)

        assert wrapper.params["required_param"] == 5
        assert wrapper.params["optional_param"] == 20
        assert wrapper.params["another_optional"] == "test"

    def test_roundtrip_with_nested(self):
        """Test that get_params/set_params roundtrip works with nested estimators."""
        inner = SimpleWrapper(simple=SimpleEstimator, required_param=1, optional_param=10)

        class ClassWithNested(BaseTestClass):
            def __init__(self, estimator=None, other_param=5):
                self.estimator = estimator
                self.other_param = other_param

        outer = SimpleWrapper(simple=ClassWithNested, estimator=inner, other_param=10)

        all_params = outer.get_params(deep=True)
        all_params.pop("simple", None)
        outer.set_params(**all_params)

        assert outer.params["estimator"].params["optional_param"] == 10
        assert outer.params["other_param"] == 10

    def test_sklearn_compatibility(self):
        """Test that get_params/set_params work with sklearn's parameter handling."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        params = wrapper.get_params()
        params["optional_param"] = 100
        params.pop("simple", None)
        wrapper.set_params(**params)

        assert wrapper.params["optional_param"] == 100


class TestCoverageCompletion:
    """Tests ensuring complete coverage of edge paths."""

    def test_validate_estimator_params_skip_nested(self):
        """Test _validate_estimator_params with validate_nested=False."""

        class TestWrapper(BaseClassWrapper):
            _estimator_name = "test"
            _estimator_base_class = BaseTestClass

            def expose_validation(self, params, validate_nested=True):
                return self._validate_estimator_params(params, validate_nested=validate_nested)

        wrapper = TestWrapper(test=SimpleEstimator, required_param=5)

        params = {"param1": "value1", "param2": "value2"}
        result = wrapper.expose_validation(params, validate_nested=False)

        assert result == {"param1": "value1", "param2": "value2"}
