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
    DefaultClassWrapper,
    MissingBaseClassWrapper,
    MissingNameWrapper,
    NoRequiredParams,
    NotBaseClass,
    SimpleEstimator,
    SimpleWrapper,
)


class TestRequiredParamValue:
    """Tests for the REQUIRED_PARAM_VALUE sentinel constant."""

    def test_is_string(self):
        """Test that REQUIRED_PARAM_VALUE is a string sentinel."""
        assert isinstance(REQUIRED_PARAM_VALUE, str)
        assert REQUIRED_PARAM_VALUE == "__REQUIRED__"


class TestWrapperInit:
    """Tests for wrapper initialization."""

    def test_with_valid_params(self):
        """Test wrapper initialization with valid parameters."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5, optional_param=20)
        assert wrapper.estimator_class == SimpleEstimator
        assert wrapper.params["required_param"] == 5
        assert wrapper.params["optional_param"] == 20

    def test_only_estimator_class(self):
        """Test initialization with only estimator_class parameter."""
        wrapper = SimpleWrapper(simple=NoRequiredParams)
        assert wrapper.estimator_class == NoRequiredParams
        assert wrapper.params["param1"] == 1
        assert wrapper.params["param2"] == "test"

    def test_missing_estimator_class(self):
        """Test that missing estimator class raises TypeError."""
        with pytest.raises(TypeError, match="missing required keyword argument: 'simple'"):
            SimpleWrapper()

    def test_inherits_base_estimator(self):
        """Test that BaseClassWrapper inherits from sklearn's BaseEstimator."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        assert isinstance(wrapper, BaseEstimator)

    def test_required_parameters_class_attribute(self):
        """Test that _required_parameters class attribute is set correctly."""
        assert SimpleWrapper._required_parameters == ["simple"]


class TestDefaultClass:
    """Tests for _estimator_default_class behavior."""

    def test_default_class_used(self):
        """Test that _estimator_default_class is used when no class is passed."""
        wrapper = DefaultClassWrapper(param1=99)
        assert wrapper.estimator_class == NoRequiredParams
        assert wrapper.params["param1"] == 99

    def test_default_class_no_params(self):
        """Test construction with default class and no extra params."""
        wrapper = DefaultClassWrapper()
        assert wrapper.estimator_class == NoRequiredParams
        assert wrapper.params["param1"] == 1
        assert wrapper.params["param2"] == "test"

    def test_default_class_override(self):
        """Test that default class can be overridden explicitly."""
        wrapper = DefaultClassWrapper(simple=SimpleEstimator, required_param=5)
        assert wrapper.estimator_class == SimpleEstimator
        assert wrapper.params["required_param"] == 5

    def test_default_class_not_required(self):
        """Test that _required_parameters is empty when a default class exists."""
        assert DefaultClassWrapper._required_parameters == []
        assert SimpleWrapper._required_parameters == ["simple"]


class TestEstimatorProperties:
    """Tests for estimator_name and estimator_base_class properties."""

    def test_estimator_name(self):
        """Test that estimator_name property returns the correct value."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        assert wrapper.estimator_name == "simple"

    def test_estimator_name_not_defined(self):
        """Test that accessing estimator_name raises error when not defined."""
        with pytest.raises(ValueError, match="Class should define a static `_estimator_name`"):
            MissingNameWrapper(simple=SimpleEstimator, required_param=5)

    def test_estimator_name_raises_when_unset(self):
        """Test that estimator_name property raises when _estimator_name is not a string."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper._estimator_name = None
        with pytest.raises(ValueError, match="Class should define a static `_estimator_name`"):
            _ = wrapper.estimator_name

    def test_estimator_base_class(self):
        """Test that estimator_base_class property returns the correct value."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        assert wrapper.estimator_base_class == BaseTestClass

    def test_estimator_base_class_not_defined(self):
        """Test that initialization raises error when base class not defined."""
        with pytest.raises(ValueError, match="Class should define a static `_estimator_base_class`"):
            MissingBaseClassWrapper(simple=SimpleEstimator, required_param=5)


class TestValidateEstimatorParams:
    """Tests for _validate_estimator_params method."""

    def test_all_provided(self):
        """Test parameter validation with all parameters provided."""
        wrapper = SimpleWrapper(
            simple=SimpleEstimator,
            required_param=5,
            optional_param=15,
            another_optional="custom",
        )
        assert wrapper.params["required_param"] == 5
        assert wrapper.params["optional_param"] == 15
        assert wrapper.params["another_optional"] == "custom"

    def test_with_defaults(self):
        """Test that default parameters are correctly filled in."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        assert wrapper.params["required_param"] == 5
        assert wrapper.params["optional_param"] == 10
        assert wrapper.params["another_optional"] == "default"

    def test_required_marked_with_sentinel(self):
        """Test that missing required parameters are marked with sentinel."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, optional_param=15)
        assert wrapper.params["required_param"] == REQUIRED_PARAM_VALUE
        assert wrapper.params["optional_param"] == 15

    def test_invalid_param(self):
        """Test that invalid parameter names raise ValueError."""
        with pytest.raises(ValueError, match="'invalid_param' is not a valid parameter for class 'SimpleEstimator'"):
            SimpleWrapper(simple=SimpleEstimator, required_param=5, invalid_param=100)

    def test_empty_params(self):
        """Test validation with no parameters provided."""
        wrapper = SimpleWrapper(simple=SimpleEstimator)
        assert wrapper.params["required_param"] == REQUIRED_PARAM_VALUE
        assert wrapper.params["optional_param"] == 10
        assert wrapper.params["another_optional"] == "default"


class TestValidateEstimatorClass:
    """Tests for _validate_params and _validate_estimator_class."""

    def test_valid_subclass(self):
        """Test that _validate_params succeeds with valid subclass."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)
        wrapper._validate_params()

    def test_invalid_subclass(self):
        """Test that _validate_params raises error with invalid subclass."""
        with pytest.raises(ValueError, match="should be derived from"):
            SimpleWrapper(simple=NotBaseClass)

    def test_base_class_itself(self):
        """Test that base class itself is valid."""
        wrapper = SimpleWrapper(simple=BaseTestClass)
        wrapper._validate_params()

    def test_not_a_class(self):
        """Test that passing a non-class raises TypeError."""
        not_a_class = SimpleEstimator(required_param=5)
        with pytest.raises(TypeError, match="is not a class"):
            SimpleWrapper(simple=not_a_class, required_param=10)


class TestInstantiate:
    """Tests for the instantiate method."""

    def test_creates_instance(self):
        """Test that instantiate creates an instance of the wrapped class."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=42, optional_param=99)
        result = wrapper.instantiate()

        assert result is wrapper
        assert hasattr(wrapper, "instance_")
        assert isinstance(wrapper.instance_, SimpleEstimator)
        assert wrapper.instance_.required_param == 42
        assert wrapper.instance_.optional_param == 99

    def test_with_defaults(self):
        """Test instantiate with default parameter values."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=7)
        wrapper.instantiate()

        assert wrapper.instance_.required_param == 7
        assert wrapper.instance_.optional_param == 10
        assert wrapper.instance_.another_optional == "default"

    def test_missing_required_param(self):
        """Test that instantiate raises error when required param is missing."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, optional_param=15)
        with pytest.raises(ValueError, match="Class 'SimpleEstimator' requires parameter 'required_param'"):
            wrapper.instantiate()

    def test_validates_subclass(self):
        """Test that instantiate calls _validate_params."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, optional_param=15)
        wrapper.estimator_class = NotBaseClass
        with pytest.raises(ValueError, match="should be derived from"):
            wrapper.instantiate()

    def test_all_params_provided(self):
        """Test instantiate when all parameters are explicitly provided."""
        wrapper = SimpleWrapper(
            simple=SimpleEstimator,
            required_param="test",
            optional_param=50,
            another_optional="modified",
        )
        wrapper.instantiate()

        assert wrapper.instance_.required_param == "test"
        assert wrapper.instance_.optional_param == 50
        assert wrapper.instance_.another_optional == "modified"

    def test_multiple_times(self):
        """Test that instantiate can be called multiple times."""
        wrapper = SimpleWrapper(simple=SimpleEstimator, required_param=5)

        instance1 = wrapper.instantiate()
        instance2 = wrapper.instantiate()

        assert hasattr(wrapper, "instance_")
        assert instance1 is instance2


class TestMiscellaneous:
    """Tests for complex class signatures and type preservation."""

    def test_wrapper_with_complex_class_signature(self):
        """Test wrapper with a class that has complex signature."""

        class ComplexClass(BaseTestClass):
            def __init__(self, a, b=1, c="test", d=None, e=None):
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.e = e if e is not None else []

        wrapper = SimpleWrapper(simple=ComplexClass, a="required_value", b=2, c="custom")

        assert wrapper.params["a"] == "required_value"
        assert wrapper.params["b"] == 2
        assert wrapper.params["c"] == "custom"
        assert wrapper.params["d"] is None
        assert wrapper.params["e"] is None

    def test_wrapper_preserves_param_types(self):
        """Test that wrapper preserves parameter types correctly."""

        class TypedClass(BaseTestClass):
            def __init__(self, int_param: int = 5, str_param: str = "default", list_param=None):
                self.int_param = int_param
                self.str_param = str_param
                self.list_param = list_param if list_param is not None else []

        wrapper = SimpleWrapper(
            simple=TypedClass,
            int_param=10,
            str_param="custom",
            list_param=[1, 2, 3],
        )
        wrapper.instantiate()

        assert wrapper.instance_.int_param == 10
        assert wrapper.instance_.str_param == "custom"
        assert wrapper.instance_.list_param == [1, 2, 3]
