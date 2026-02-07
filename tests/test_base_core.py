"""Core functionality tests for BaseClassWrapper.

Tests:
- Initialization and constructor validation
- Property accessors (estimator_name, estimator_base_class)
- Instance creation (instantiate method)
- Internal validation methods
- REQUIRED_PARAM_VALUE constant
"""

import pytest
from sklearn.base import BaseEstimator

from sklearn_wrap.base import REQUIRED_PARAM_VALUE

from .conftest import (
    BaseTestClass,
    MissingBaseClassWrapper,
    MissingNameWrapper,
    NoRequiredParams,
    NotBaseClass,
    SimpleEstimator,
    SimpleWrapper,
)

# ============================================================================
# REQUIRED_PARAM_VALUE Constant Tests
# ============================================================================


def test_required_param_value_constant():
    """Test that REQUIRED_PARAM_VALUE is a string sentinel."""
    assert isinstance(REQUIRED_PARAM_VALUE, str)
    assert REQUIRED_PARAM_VALUE == "__REQUIRED__"


# ============================================================================
# Initialization Tests
# ============================================================================


def test_wrapper_init_with_valid_params():
    """Test wrapper initialization with valid parameters."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, optional_param=20)
    assert wrapper.estimator_class == SimpleEstimator
    assert wrapper.params["required_param"] == 5
    assert wrapper.params["optional_param"] == 20


def test_wrapper_init_only_estimator_class():
    """Test initialization with only estimator_class parameter."""
    wrapper = SimpleWrapper(estimator_class=NoRequiredParams)
    assert wrapper.estimator_class == NoRequiredParams
    assert wrapper.params["param1"] == 1
    assert wrapper.params["param2"] == "test"


def test_wrapper_init_missing_estimator_class():
    """Test that missing estimator_class raises TypeError."""
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        SimpleWrapper()


def test_wrapper_inherits_base_estimator():
    """Test that BaseClassWrapper inherits from sklearn's BaseEstimator."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    assert isinstance(wrapper, BaseEstimator)


# ============================================================================
# Property Tests
# ============================================================================


def test_estimator_name_property():
    """Test that estimator_name property returns the correct value."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    assert wrapper.estimator_name == "simple"


def test_estimator_name_not_defined():
    """Test that accessing estimator_name raises error when not defined."""
    wrapper = MissingNameWrapper(estimator_class=SimpleEstimator, required_param=5)
    with pytest.raises(ValueError, match="Class should define a static `_estimator_name`"):
        _ = wrapper.estimator_name


def test_estimator_base_class_property():
    """Test that estimator_base_class property returns the correct value."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    assert wrapper.estimator_base_class == BaseTestClass


def test_estimator_base_class_not_defined():
    """Test that initialization raises error when base class not defined."""
    with pytest.raises(ValueError, match="Class should define a static `_estimator_base_class`"):
        MissingBaseClassWrapper(estimator_class=SimpleEstimator, required_param=5)


# ============================================================================
# Parameter Validation Tests (_validate_estimator_params)
# ============================================================================


def test_validate_estimator_params_all_provided():
    """Test parameter validation with all parameters provided."""
    wrapper = SimpleWrapper(
        estimator_class=SimpleEstimator,
        required_param=5,
        optional_param=15,
        another_optional="custom",
    )
    assert wrapper.params["required_param"] == 5
    assert wrapper.params["optional_param"] == 15
    assert wrapper.params["another_optional"] == "custom"


def test_validate_estimator_params_with_defaults():
    """Test that default parameters are correctly filled in."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    assert wrapper.params["required_param"] == 5
    assert wrapper.params["optional_param"] == 10
    assert wrapper.params["another_optional"] == "default"


def test_validate_estimator_params_required_marked():
    """Test that missing required parameters are marked with sentinel."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, optional_param=15)
    assert wrapper.params["required_param"] == REQUIRED_PARAM_VALUE
    assert wrapper.params["optional_param"] == 15


def test_validate_estimator_params_invalid_param():
    """Test that invalid parameter names raise ValueError."""
    with pytest.raises(ValueError, match="'invalid_param' is not a valid parameter for class 'SimpleEstimator'"):
        SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, invalid_param=100)


def test_validate_estimator_params_empty():
    """Test validation with no parameters provided."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator)
    assert wrapper.params["required_param"] == REQUIRED_PARAM_VALUE
    assert wrapper.params["optional_param"] == 10
    assert wrapper.params["another_optional"] == "default"


# ============================================================================
# Estimator Class Validation Tests (_validate_params)
# ============================================================================


def test_validate_params_valid_subclass():
    """Test that _validate_params succeeds with valid subclass."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)
    wrapper._validate_params()  # Should not raise


def test_validate_params_invalid_subclass():
    """Test that _validate_params raises error with invalid subclass."""
    with pytest.raises(ValueError, match="should be derived from"):
        SimpleWrapper(estimator_class=NotBaseClass)


def test_validate_params_base_class_itself():
    """Test that base class itself is valid."""
    wrapper = SimpleWrapper(estimator_class=BaseTestClass)
    wrapper._validate_params()  # Should not raise


def test_validate_params_not_a_class():
    """Test that passing a non-class raises TypeError."""
    # Try to pass an instance instead of a class
    not_a_class = SimpleEstimator(required_param=5)

    with pytest.raises(TypeError, match="is not a class"):
        SimpleWrapper(estimator_class=not_a_class, required_param=10)


# ============================================================================
# Instantiate Tests
# ============================================================================


def test_instantiate_creates_instance():
    """Test that instantiate creates an instance of the wrapped class."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=42, optional_param=99)
    result = wrapper.instantiate()

    assert result is wrapper
    assert hasattr(wrapper, "instance_")
    assert isinstance(wrapper.instance_, SimpleEstimator)
    assert wrapper.instance_.required_param == 42
    assert wrapper.instance_.optional_param == 99


def test_instantiate_with_defaults():
    """Test instantiate with default parameter values."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=7)
    wrapper.instantiate()

    assert wrapper.instance_.required_param == 7
    assert wrapper.instance_.optional_param == 10
    assert wrapper.instance_.another_optional == "default"


def test_instantiate_missing_required_param():
    """Test that instantiate raises error when required param is missing."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, optional_param=15)
    with pytest.raises(ValueError, match="Class 'SimpleEstimator' requires parameter 'required_param'"):
        wrapper.instantiate()


def test_instantiate_validates_subclass():
    """Test that instantiate calls _validate_params."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, optional_param=15)
    wrapper.estimator_class = NotBaseClass  # Bypass __init__ check
    with pytest.raises(ValueError, match="should be derived from"):
        wrapper.instantiate()


def test_instantiate_all_params_provided():
    """Test instantiate when all parameters are explicitly provided."""
    wrapper = SimpleWrapper(
        estimator_class=SimpleEstimator,
        required_param="test",
        optional_param=50,
        another_optional="modified",
    )
    wrapper.instantiate()

    assert wrapper.instance_.required_param == "test"
    assert wrapper.instance_.optional_param == 50
    assert wrapper.instance_.another_optional == "modified"


def test_instantiate_multiple_times():
    """Test that instantiate can be called multiple times."""
    wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5)

    instance1 = wrapper.instantiate()
    instance2 = wrapper.instantiate()

    # Should return same wrapper instance (self)
    assert hasattr(wrapper, "instance_")
    assert instance1 is instance2  # Both should be the wrapper itself


# ============================================================================
# Edge Cases
# ============================================================================


def test_wrapper_with_complex_class_signature():
    """Test wrapper with a class that has complex signature."""

    class ComplexClass(BaseTestClass):
        def __init__(self, a, b=1, c="test", d=None, e=None):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.e = e if e is not None else []

    wrapper = SimpleWrapper(estimator_class=ComplexClass, a="required_value", b=2, c="custom")

    assert wrapper.params["a"] == "required_value"
    assert wrapper.params["b"] == 2
    assert wrapper.params["c"] == "custom"
    assert wrapper.params["d"] is None
    assert wrapper.params["e"] is None


def test_wrapper_preserves_param_types():
    """Test that wrapper preserves parameter types correctly."""

    class TypedClass(BaseTestClass):
        def __init__(self, int_param: int = 5, str_param: str = "default", list_param=None):
            self.int_param = int_param
            self.str_param = str_param
            self.list_param = list_param if list_param is not None else []

    wrapper = SimpleWrapper(
        estimator_class=TypedClass,
        int_param=10,
        str_param="custom",
        list_param=[1, 2, 3],
    )
    wrapper.instantiate()

    assert wrapper.instance_.int_param == 10
    assert wrapper.instance_.str_param == "custom"
    assert wrapper.instance_.list_param == [1, 2, 3]


def test_required_parameters_class_attribute():
    """Test that _required_parameters class attribute is set correctly."""
    assert SimpleWrapper._required_parameters == ["estimator_class"]
