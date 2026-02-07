"""Edge case tests for BaseClassWrapper.

Tests special scenarios:
- 4+ level nesting
- Classes with __slots__
- Property-based parameters
- Dynamically created classes
- Parameter dict mutation safety
- Classes with callable defaults
- Abstract base classes
"""

from abc import ABC, abstractmethod

import pytest

from .conftest import BaseTestClass, SimpleEstimator, SimpleWrapper

# ============================================================================
# Deep Nesting Tests (4+ levels)
# ============================================================================


class TestDeepNesting:
    """Test very deep nested wrapper structures."""

    def test_four_level_nesting(self):
        """Test 4-level nested wrapper structure."""

        class ClassWithNested(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        level1 = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1)
        level2 = SimpleWrapper(estimator_class=ClassWithNested, inner=level1)
        level3 = SimpleWrapper(estimator_class=ClassWithNested, inner=level2)
        level4 = SimpleWrapper(estimator_class=ClassWithNested, inner=level3)

        # Get deep params
        params = level4.get_params(deep=True)

        # Verify deep nesting is captured
        assert "inner__inner__inner__required_param" in params
        assert params["inner__inner__inner__required_param"] == 1

        # Set deep nested parameter
        level4.set_params(inner__inner__inner__optional_param=999)

        # Verify it propagated all the way down
        assert level1.params["optional_param"] == 999

    def test_five_level_nesting(self):
        """Test 5-level nested wrapper structure."""

        class ClassWithNested(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        level1 = SimpleWrapper(estimator_class=SimpleEstimator, required_param=1, optional_param=10)
        level2 = SimpleWrapper(estimator_class=ClassWithNested, inner=level1)
        level3 = SimpleWrapper(estimator_class=ClassWithNested, inner=level2)
        level4 = SimpleWrapper(estimator_class=ClassWithNested, inner=level3)
        level5 = SimpleWrapper(estimator_class=ClassWithNested, inner=level4)

        # Set deeply nested parameter
        level5.set_params(inner__inner__inner__inner__optional_param=555)

        # Verify
        assert level1.params["optional_param"] == 555


# ============================================================================
# Classes with __slots__
# ============================================================================


class TestSlotsClasses:
    """Test wrapping classes with __slots__."""

    def test_wrap_class_with_slots(self):
        """Test wrapping a class that uses __slots__."""

        class SlottedClass(BaseTestClass):
            __slots__ = ("param1", "param2")

            def __init__(self, param1, param2=10):
                self.param1 = param1
                self.param2 = param2

        wrapper = SimpleWrapper(estimator_class=SlottedClass, param1=5, param2=20)

        # Should work normally
        assert wrapper.params["param1"] == 5
        assert wrapper.params["param2"] == 20

        # Instantiate
        wrapper.instantiate()
        assert wrapper.instance_.param1 == 5
        assert wrapper.instance_.param2 == 20

    def test_wrap_class_with_slots_and_dict(self):
        """Test wrapping class with both __slots__ and __dict__."""

        class MixedSlottedClass(BaseTestClass):
            __slots__ = ("fixed_param",)

            def __init__(self, fixed_param, dynamic_param=None):
                self.fixed_param = fixed_param
                if dynamic_param is not None:
                    self.__dict__["dynamic_param"] = dynamic_param

        # Note: This class has __slots__ but also allows __dict__ for additional attrs
        wrapper = SimpleWrapper(estimator_class=MixedSlottedClass, fixed_param=10, dynamic_param="test")

        wrapper.instantiate()
        assert wrapper.instance_.fixed_param == 10


# ============================================================================
# Property-Based Parameters
# ============================================================================


class TestPropertyBasedParams:
    """Test wrapping classes that use properties."""

    def test_wrap_class_with_property_setter(self):
        """Test wrapping class that uses property setters."""

        class PropertyClass(BaseTestClass):
            def __init__(self, value=10):
                self._value = None
                self.value = value  # Use property setter

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, val):
                self._value = val * 2  # Property transforms value

        wrapper = SimpleWrapper(estimator_class=PropertyClass, value=5)

        wrapper.instantiate()
        # Value should be transformed by property setter
        assert wrapper.instance_.value == 10  # 5 * 2

    def test_wrap_class_with_readonly_property(self):
        """Test wrapping class with read-only properties."""

        class ReadOnlyPropertyClass(BaseTestClass):
            def __init__(self, base_value=10):
                self.base_value = base_value

            @property
            def computed_value(self):
                return self.base_value * 2

        wrapper = SimpleWrapper(estimator_class=ReadOnlyPropertyClass, base_value=15)

        wrapper.instantiate()
        assert wrapper.instance_.base_value == 15
        assert wrapper.instance_.computed_value == 30


# ============================================================================
# Dynamically Created Classes
# ============================================================================


class TestDynamicClasses:
    """Test wrapping dynamically created classes."""

    def test_wrap_class_created_with_type(self):
        """Test wrapping a class created dynamically with type()."""

        def __init__(self, param1, param2=20):
            self.param1 = param1
            self.param2 = param2

        # Create class dynamically
        DynamicClass = type("DynamicClass", (BaseTestClass,), {"__init__": __init__})

        wrapper = SimpleWrapper(estimator_class=DynamicClass, param1=10, param2=30)

        assert wrapper.params["param1"] == 10
        assert wrapper.params["param2"] == 30

        wrapper.instantiate()
        assert wrapper.instance_.param1 == 10
        assert wrapper.instance_.param2 == 30

    def test_wrap_class_with_dynamic_methods(self):
        """Test wrapping class with dynamically added methods."""

        class BaseClass(BaseTestClass):
            def __init__(self, value=10):
                self.value = value

        # Dynamically add a method
        def custom_method(self):
            return self.value * 2

        BaseClass.custom_method = custom_method

        wrapper = SimpleWrapper(estimator_class=BaseClass, value=15)
        wrapper.instantiate()

        # Dynamically added method should work
        assert wrapper.instance_.custom_method() == 30


# ============================================================================
# Parameter Dict Mutation Safety
# ============================================================================


class TestParamDictMutationSafety:
    """Test that modifying parameter dicts doesn't affect wrapper."""

    def test_external_dict_mutation_doesnt_affect_wrapper(self):
        """Test that modifying external dict doesn't affect wrapper params."""
        params_dict = {"required_param": 5, "optional_param": 20}

        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, **params_dict)

        # Modify external dict
        params_dict["optional_param"] = 999
        params_dict["new_param"] = "should not appear"

        # Wrapper params should be unchanged
        assert wrapper.params["optional_param"] == 20
        assert "new_param" not in wrapper.params

    def test_get_params_dict_mutation_doesnt_affect_wrapper(self):
        """Test that modifying returned params dict doesn't affect wrapper."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, optional_param=20)

        # Get params
        params = wrapper.get_params()

        # Modify returned dict
        params["optional_param"] = 999

        # Wrapper should be unchanged
        assert wrapper.params["optional_param"] == 20

    def test_params_attribute_mutation_safety(self):
        """Test that wrapper.params modifications are handled correctly."""
        wrapper = SimpleWrapper(estimator_class=SimpleEstimator, required_param=5, optional_param=20)

        # Direct modification of params dict (not recommended but testing safety)
        wrapper.params["optional_param"] = 999

        # This does modify the wrapper (it's a mutable dict)
        assert wrapper.params["optional_param"] == 999

        # But validation happens on instantiate or set_params
        wrapper.set_params(optional_param=20)
        assert wrapper.params["optional_param"] == 20


# ============================================================================
# Callable Defaults
# ============================================================================


class TestCallableDefaults:
    """Test classes with callable default parameters."""

    def test_wrap_class_with_lambda_default(self):
        """Test wrapping class with lambda as default value."""

        class ClassWithCallableDefault(BaseTestClass):
            def __init__(self, value=10, factory=None):
                self.value = value
                self.factory = factory if factory is not None else (lambda: [])

        wrapper = SimpleWrapper(estimator_class=ClassWithCallableDefault, value=15)

        wrapper.instantiate()
        assert wrapper.instance_.value == 15
        assert callable(wrapper.instance_.factory)

    def test_wrap_class_with_function_default(self):
        """Test wrapping class with function as default."""

        def default_function():
            return {"key": "value"}

        class ClassWithFunctionDefault(BaseTestClass):
            def __init__(self, value=10, func=default_function):
                self.value = value
                self.func = func

        wrapper = SimpleWrapper(estimator_class=ClassWithFunctionDefault, value=20)

        wrapper.instantiate()
        assert wrapper.instance_.value == 20
        assert wrapper.instance_.func == default_function


# ============================================================================
# Abstract Base Classes
# ============================================================================


class TestAbstractBaseClasses:
    """Test wrapping abstract base classes (should fail appropriately)."""

    def test_wrap_abstract_class_fails_on_instantiate(self):
        """Test that wrapping ABC fails during instantiate (not __init__)."""

        class AbstractClass(BaseTestClass, ABC):
            @abstractmethod
            def abstract_method(self):
                pass

            def __init__(self, value=10):
                self.value = value

        # Creating wrapper should work (abstract check happens at instantiation)
        wrapper = SimpleWrapper(estimator_class=AbstractClass, value=15)

        # But instantiating should fail
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            wrapper.instantiate()

    def test_wrap_concrete_subclass_of_abc(self):
        """Test wrapping concrete implementation of ABC works."""

        class AbstractClass(BaseTestClass, ABC):
            @abstractmethod
            def abstract_method(self):
                pass

            def __init__(self, value=10):
                self.value = value

        class ConcreteClass(AbstractClass):
            def abstract_method(self):
                return self.value * 2

        wrapper = SimpleWrapper(estimator_class=ConcreteClass, value=15)

        wrapper.instantiate()
        assert wrapper.instance_.value == 15
        assert wrapper.instance_.abstract_method() == 30


# ============================================================================
# Multiple Inheritance
# ============================================================================


class TestMultipleInheritance:
    """Test wrapping classes with multiple inheritance."""

    def test_wrap_class_with_multiple_inheritance(self):
        """Test wrapping class that inherits from multiple bases."""

        class Mixin:
            def mixin_method(self):
                return "mixin"

        class MultipleInheritanceClass(BaseTestClass, Mixin):
            def __init__(self, value=10):
                self.value = value

        wrapper = SimpleWrapper(estimator_class=MultipleInheritanceClass, value=25)

        wrapper.instantiate()
        assert wrapper.instance_.value == 25
        assert wrapper.instance_.mixin_method() == "mixin"

    def test_wrap_class_with_mro_complexity(self):
        """Test wrapping class with complex method resolution order."""

        class Base1(BaseTestClass):
            def method(self):
                return "base1"

        class Base2:
            def method(self):
                return "base2"

        class ComplexMRO(Base1, Base2):
            def __init__(self, value=10):
                self.value = value

        wrapper = SimpleWrapper(estimator_class=ComplexMRO, value=30)

        wrapper.instantiate()
        # Should use Base1's method (first in MRO)
        assert wrapper.instance_.method() == "base1"


# ============================================================================
# Special Parameter Names
# ============================================================================


class TestSpecialParameterNames:
    """Test handling of special parameter names."""

    def test_wrap_class_with_kwargs_parameter(self):
        """Test wrapping class that accepts **kwargs."""

        class ClassWithKwargs(BaseTestClass):
            def __init__(self, required, **kwargs):
                self.required = required
                self.kwargs = kwargs

        wrapper = SimpleWrapper(estimator_class=ClassWithKwargs, required=5, extra1="value1", extra2="value2")

        wrapper.instantiate()
        assert wrapper.instance_.required == 5
        assert wrapper.instance_.kwargs["extra1"] == "value1"
        assert wrapper.instance_.kwargs["extra2"] == "value2"

    def test_wrap_class_with_args_parameter(self):
        """Test wrapping class that accepts *args."""

        class ClassWithArgs(BaseTestClass):
            def __init__(self, *args):
                self.args = args

        # Note: *args in constructor signature can't be easily filled from kwargs
        # This tests that the wrapper handles the signature correctly
        wrapper = SimpleWrapper(estimator_class=ClassWithArgs)

        # Should instantiate with no args
        wrapper.instantiate()
        assert wrapper.instance_.args == ()


# ============================================================================
# Edge Case Combinations
# ============================================================================


class TestEdgeCaseCombinations:
    """Test combinations of edge cases."""

    def test_deep_nesting_with_properties(self):
        """Test deep nesting combined with property-based classes."""

        class PropertyClass(BaseTestClass):
            def __init__(self, value=10):
                self._value = value

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, val):
                self._value = val

        class ClassWithNested(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        level1 = SimpleWrapper(estimator_class=PropertyClass, value=5)
        level2 = SimpleWrapper(estimator_class=ClassWithNested, inner=level1)
        level3 = SimpleWrapper(estimator_class=ClassWithNested, inner=level2)

        # Set nested parameter
        level3.set_params(inner__inner__value=100)

        # Verify
        assert level1.params["value"] == 100

    def test_slots_with_nested_wrappers(self):
        """Test __slots__ classes in nested wrapper structure."""

        class SlottedClass(BaseTestClass):
            __slots__ = ("value",)

            def __init__(self, value=10):
                self.value = value

        class ClassWithNested(BaseTestClass):
            def __init__(self, inner=None):
                self.inner = inner

        inner = SimpleWrapper(estimator_class=SlottedClass, value=50)
        outer = SimpleWrapper(estimator_class=ClassWithNested, inner=inner)

        # Get nested params
        params = outer.get_params(deep=True)
        assert params["inner__value"] == 50

        # Set nested params
        outer.set_params(inner__value=75)
        assert inner.params["value"] == 75
