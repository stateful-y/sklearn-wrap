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

# ============================================================================
# get_params Tests
# ============================================================================


def test_get_params_returns_dict():
    """Test that get_params returns a dictionary."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    params = wrapper.get_params()
    assert isinstance(params, dict)


def test_get_params_includes_estimator_name():
    """Test that get_params includes the estimator name key."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    params = wrapper.get_params()
    assert "simple" in params
    assert params["simple"] == SimpleEstimator


def test_get_params_includes_all_params():
    """Test that get_params includes all constructor parameters."""
    wrapper = SimpleWrapper(
        estimator_class=SimpleEstimator,
        required_param=5,
        optional_param=20,
        another_optional="test",
    )
    params = wrapper.get_params()

    assert params["required_param"] == 5
    assert params["optional_param"] == 20
    assert params["another_optional"] == "test"


def test_get_params_includes_defaults():
    """Test that get_params includes default parameter values."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    params = wrapper.get_params()

    assert params["optional_param"] == 10
    assert params["another_optional"] == "default"


def test_get_params_with_deep_parameter():
    """Test that get_params accepts deep parameter."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    params_deep = wrapper.get_params(deep=True)
    params_shallow = wrapper.get_params(deep=False)

    # Both should include basic params
    assert "required_param" in params_deep
    assert "required_param" in params_shallow


def test_get_params_deep_with_nested_estimator():
    """Test get_params with deep=True and nested estimators."""
    inner_wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1)

    class ClassWithNestedEstimator(BaseTestClass):
        def __init__(self, nested=None, simple_param=5):
            self.nested = nested
            self.simple_param = simple_param

    wrapper = SimpleWrapper(
        estimator_class=ClassWithNestedEstimator,
        nested=inner_wrapper,
        simple_param=10,
    )

    # Get params with deep=True should include nested estimator params
    params = wrapper.get_params(deep=True)
    assert "nested" in params
    assert "simple_param" in params
    assert params["nested"] == inner_wrapper
    # Should include nested params with __ prefix
    assert "nested__required_param" in params
    assert params["nested__required_param"] == 1


def test_get_params_deep_with_non_wrapper_nested():
    """Test get_params with deep=True when nested object has get_params but isn't a wrapper."""

    class NonWrapperWithGetParams:
        """A class with get_params but not a BaseClassWrapper."""

        def __init__(self, value=42):
            self.value = value

        def get_params(self, deep=True):
            return {"value": self.value}

    # Manually set a param to this object
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    wrapper.params["custom_obj"] = NonWrapperWithGetParams(value=100)

    # Get params with deep=True - should include nested params
    params = wrapper.get_params(deep=True)
    assert "custom_obj__value" in params
    assert params["custom_obj__value"] == 100


# ============================================================================
# set_params Tests
# ============================================================================


def test_set_params_updates_params():
    """Test that set_params updates parameters correctly."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    wrapper.set_params(optional_param=30)

    assert wrapper.params["optional_param"] == 30


def test_set_params_returns_self():
    """Test that set_params returns self for chaining."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    result = wrapper.set_params(optional_param=30)

    assert result is wrapper


def test_set_params_multiple_params():
    """Test setting multiple parameters at once."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    wrapper.set_params(optional_param=25, another_optional="new")

    assert wrapper.params["optional_param"] == 25
    assert wrapper.params["another_optional"] == "new"


def test_set_params_empty():
    """Test set_params with no arguments."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    original_class = wrapper.estimator_class

    wrapper.set_params()

    # Estimator class should remain unchanged
    assert wrapper.estimator_class == original_class
    # Parameters are revalidated
    assert "required_param" in wrapper.params
    assert "optional_param" in wrapper.params


def test_set_params_then_instantiate():
    """Test changing params and then instantiating."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    wrapper.set_params(required_param=5, optional_param=77)
    wrapper.instantiate()

    assert wrapper.instance_.required_param == 5
    assert wrapper.instance_.optional_param == 77


def test_set_params_changes_estimator_class():
    """Test that set_params raises error when trying to change the estimator class."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    with pytest.raises(ValueError, match="Cannot change estimator class via set_params"):
        wrapper.set_params(simple=NoRequiredParams)


def test_set_params_validates_new_params():
    """Test that set_params validates new parameters."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    with pytest.raises(ValueError, match="'invalid_param' is not a valid parameter for class 'SimpleEstimator'"):
        wrapper.set_params(invalid_param=100)


def test_set_params_with_estimator_class_key():
    """Test that set_params rejects estimator_class parameter."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    with pytest.raises(ValueError, match="Cannot change estimator class via set_params"):
        wrapper.set_params(estimator_class=NoRequiredParams)


def test_set_params_with_non_type_estimator_value():
    """Test that set_params raises error when trying to set estimator name."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Passing any value for estimator name should raise an error
    with pytest.raises(ValueError, match="Cannot change estimator class via set_params"):
        wrapper.set_params(simple="not_a_type")


def test_set_params_invalid_param_after_validation():
    """Test error when param doesn't exist after validation."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    with pytest.raises(
        ValueError, match="'totally_invalid_param' is not a valid parameter for class 'SimpleEstimator'"
    ):
        wrapper.set_params(totally_invalid_param=100)


# ============================================================================
# Nested Parameter Tests (set_params with __ syntax)
# ============================================================================


def test_set_params_nested_basic():
    """Test basic nested parameter setting with __ syntax."""
    inner_wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None, other_param=5):
            self.estimator = estimator
            self.other_param = other_param

    outer_wrapper = SimpleWrapper(estimator_class=ClassWithNested, estimator=inner_wrapper, other_param=10)

    # Set nested parameter using __ syntax
    outer_wrapper.set_params(estimator__optional_param=100)

    # Verify the nested estimator's parameter was updated
    assert outer_wrapper.params["estimator"].params["optional_param"] == 100
    assert outer_wrapper.params["other_param"] == 10  # Other params unchanged


def test_set_params_nested_multiple_levels():
    """Test multi-level nested parameter setting."""
    # Create a 3-level nested structure
    level1 = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)

    class ClassWithNested(BaseTestClass):
        def __init__(self, inner=None):
            self.inner = inner

    level2 = SimpleWrapper(estimator_class=ClassWithNested, inner=level1)
    level3 = SimpleWrapper(estimator_class=ClassWithNested, inner=level2)

    # Set deeply nested parameter
    level3.set_params(inner__inner__optional_param=999)

    # Verify it reached the deepest level
    assert level3.params["inner"].params["inner"].params["optional_param"] == 999


def test_set_params_nested_and_simple_mixed():
    """Test setting both nested and simple parameters together."""
    inner = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None, other_param=5):
            self.estimator = estimator
            self.other_param = other_param

    outer = SimpleWrapper(estimator_class=ClassWithNested, estimator=inner, other_param=10)

    # Set both nested and simple params in one call
    outer.set_params(estimator__optional_param=200, other_param=20)

    assert outer.params["estimator"].params["optional_param"] == 200
    assert outer.params["other_param"] == 20


def test_set_params_nested_estimator_object():
    """Test set_params with an estimator object that has get_params/set_params."""
    inner_wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)

    class ClassWithEstimatorParam(BaseTestClass):
        def __init__(self, estimator=None, other_param=5):
            self.estimator = estimator
            self.other_param = other_param

    outer_wrapper = SimpleWrapper(estimator_class=ClassWithEstimatorParam, estimator=inner_wrapper, other_param=10)

    # Update regular params (not nested with __)
    outer_wrapper.set_params(other_param=20)
    assert outer_wrapper.params["other_param"] == 20

    # The estimator param contains an object with set_params
    assert outer_wrapper.params["estimator"] == inner_wrapper


def test_set_params_nested_without_set_params_method():
    """Test error when trying to set nested params on object without set_params."""

    class ClassWithScalar(BaseTestClass):
        def __init__(self, scalar_param=5):
            self.scalar_param = scalar_param

    wrapper = SimpleWrapper(estimator_class=ClassWithScalar, scalar_param=10)

    # Try to set nested parameter on a scalar value
    with pytest.raises(AttributeError, match="does not have a set_params method"):
        wrapper.set_params(scalar_param__something=100)


def test_set_params_nested_invalid_base_key():
    """Test error when trying to set nested param on non-existent base param."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Try to set a nested parameter on a base param that doesn't exist
    with pytest.raises(ValueError, match="Invalid parameter 'nested' for estimator"):
        wrapper.set_params(nested__optional_param=100)


# ============================================================================
# Parameter Constraints Tests
# ============================================================================


def test_parameter_constraints_wrapper_base_class():
    """Test parameter constraints for nested wrapper validation."""

    class SpecializedWrapper(SimpleWrapper):
        _parameter_constraints = {"estimator": [{"wrapper_base_class": BaseTestClass}]}

    # Valid: wrapper with correct base class
    inner = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1)

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None):
            self.estimator = estimator

    outer = SpecializedWrapper(estimator_class=ClassWithNested, estimator=inner)
    # This should work - no error
    outer.set_params(estimator__optional_param=100)


def test_parameter_constraints_not_wrapper():
    """Test parameter constraints reject non-wrapper values."""

    class SpecializedWrapper(SimpleWrapper):
        _parameter_constraints = {"estimator": [{"wrapper_base_class": BaseTestClass}]}

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None):
            self.estimator = estimator

    # Try to set a non-wrapper value when wrapper is required
    with pytest.raises(TypeError, match="must be a BaseClassWrapper instance"):
        SpecializedWrapper(estimator_class=ClassWithNested, estimator="not a wrapper")


def test_parameter_constraints_wrong_base_class():
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

    # Create a wrapper with wrong base class
    wrong_inner = OtherWrapper(estimator_class=OtherDummyClass, param=1)

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None):
            self.estimator = estimator

    # Should raise error about wrong base class
    with pytest.raises(ValueError, match="must wrap an estimator class derived from"):
        SpecializedWrapper(estimator_class=ClassWithNested, estimator=wrong_inner)


def test_parameter_constraints_empty():
    """Test that wrapper works with no parameter constraints."""

    class WrapperNoConstraints(SimpleWrapper):
        _parameter_constraints = {}

    # Should work fine with any parameter
    wrapper = WrapperNoConstraints(estimator_class=SimpleEstimator, required_param=5)
    assert wrapper.params["required_param"] == 5


def test_parameter_constraints_non_matching():
    """Test parameter constraint that doesn't match actual parameter."""

    class WrapperWithNonMatchingConstraint(SimpleWrapper):
        _parameter_constraints = {"nonexistent_param": [{"wrapper_base_class": BaseTestClass}]}

    # Should still work - constraints only apply if the parameter exists
    wrapper = WrapperWithNonMatchingConstraint(estimator_class=SimpleEstimator, required_param=5)
    assert wrapper.params["required_param"] == 5


# ============================================================================
# Double Underscore Validation Tests
# ============================================================================


def test_parameter_names_cannot_contain_double_underscore():
    """Test that parameter names cannot contain __ which is reserved for nested syntax."""

    class ClassWithInvalidParam(BaseTestClass):
        def __init__(self, invalid__param=5):
            self.invalid__param = invalid__param

    # Should raise error when trying to create wrapper with param containing __
    with pytest.raises(ValueError, match="cannot contain '__'"):
        SimpleWrapper(estimator_class=ClassWithInvalidParam, invalid__param=10)


def test_validate_double_underscore_directly():
    """Test direct validation of params dict with __ in name."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Direct test: try to validate params dict with __ in name
    with pytest.raises(ValueError, match="cannot contain '__'"):
        wrapper._validate_estimator_params({"some__invalid__param": 100})


# ============================================================================
# Roundtrip Compatibility Tests
# ============================================================================


def test_get_set_params_roundtrip():
    """Test that get_params/set_params roundtrip works correctly."""
    wrapper = SimpleWrapper(
        estimator_class=SimpleEstimator,
        required_param=5,
        optional_param=20,
        another_optional="test",
    )

    # Get all params
    params = wrapper.get_params()

    # Remove estimator class parameters (can't be changed via set_params)
    params.pop("simple", None)
    params.pop("estimator_class", None)

    # Set them back
    wrapper.set_params(**params)

    # Verify values are correct
    assert wrapper.params["required_param"] == 5
    assert wrapper.params["optional_param"] == 20
    assert wrapper.params["another_optional"] == "test"


def test_get_set_params_roundtrip_with_nested():
    """Test that get_params/set_params roundtrip works with nested estimators."""
    inner = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)

    class ClassWithNested(BaseTestClass):
        def __init__(self, estimator=None, other_param=5):
            self.estimator = estimator
            self.other_param = other_param

    outer = SimpleWrapper(estimator_class=ClassWithNested, estimator=inner, other_param=10)

    # Get all params (including nested)
    all_params = outer.get_params(deep=True)

    # Remove the estimator class params (can't be changed via set_params)
    all_params.pop("simple", None)
    all_params.pop("estimator_class", None)

    # Set them back - should work without error
    outer.set_params(**all_params)

    # Verify values are correct
    assert outer.params["estimator"].params["optional_param"] == 10
    assert outer.params["other_param"] == 10


def test_sklearn_get_set_params_compatibility():
    """Test that get_params/set_params work with sklearn's parameter handling."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Get params
    params = wrapper.get_params()

    # Modify params
    params["optional_param"] = 100

    # Remove estimator class parameters before setting
    params.pop("simple", None)
    params.pop("estimator_class", None)

    # Set params back
    wrapper.set_params(**params)

    assert wrapper.params["optional_param"] == 100


# ============================================================================
# Coverage Completion Tests
# ============================================================================


def test_validate_estimator_params_skip_nested_validation():
    """Test _validate_estimator_params with validate_nested=False."""

    class TestWrapper(BaseClassWrapper):
        _estimator_name = "test"
        _estimator_base_class = BaseTestClass

        def expose_validation(self, params, validate_nested=True):
            return self._validate_estimator_params(params, validate_nested=validate_nested)

    wrapper = TestWrapper(estimator_class=SimpleEstimator, required_param=5)

    # Call with validate_nested=False - should return params as-is
    params = {"param1": "value1", "param2": "value2"}
    result = wrapper.expose_validation(params, validate_nested=False)

    # Should return the params dict without validation
    assert result == {"param1": "value1", "param2": "value2"}


def test_parameter_constraint_with_non_wrapper_class():
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

    # This should raise TypeError because we're passing a class, not a BaseClassWrapper instance
    with pytest.raises(TypeError, match="must be a BaseClassWrapper instance"):
        StrictWrapper(estimator_class=OuterClass, inner=InnerClass)


def test_parameter_constraint_with_non_dict_constraint():
    """Test that non-dict constraints are skipped in _validate_nested_wrapper_param."""

    class ClassWithParam(BaseTestClass):
        def __init__(self, param=None):
            self.param = param

    class TestWrapper(BaseClassWrapper):
        _estimator_name = "test"
        _estimator_base_class = BaseTestClass
        # Non-dict constraint (e.g., a type constraint)
        _parameter_constraints = {"param": ["not_a_dict"]}

    # Should work fine - non-dict constraints are ignored by _validate_nested_wrapper_param
    wrapper = TestWrapper(estimator_class=ClassWithParam, param="value")
    assert wrapper.params["param"] == "value"


def test_set_params_with_none_value_and_constraints():
    """Test set_params with None value when constraints are defined."""

    class TestWrapper(BaseClassWrapper):
        _estimator_name = "test"
        _estimator_base_class = BaseTestClass
        _parameter_constraints = {"optional": [{"wrapper_base_class": BaseTestClass}]}

    class OuterClass(BaseTestClass):
        def __init__(self, optional=None):
            self.optional = optional

    # Create wrapper with None value
    wrapper = TestWrapper(estimator_class=OuterClass, optional=None)

    # Set to None explicitly - should skip validation
    wrapper.set_params(optional=None)
    assert wrapper.params["optional"] is None
