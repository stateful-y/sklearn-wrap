"""Shared pytest fixtures and helper classes for sklearn-wrap tests."""

import pytest

from sklearn_wrap.base import BaseClassWrapper

# ============================================================================
# Base Test Classes
# ============================================================================


class BaseTestClass:
    """Base class for test estimators."""

    pass


# ============================================================================
# Dummy Estimator Classes
# ============================================================================


class SimpleEstimator(BaseTestClass):
    """Simple estimator with required and optional parameters."""

    def __init__(self, required_param, optional_param=10, another_optional="default"):
        self.required_param = required_param
        self.optional_param = optional_param
        self.another_optional = another_optional


class NoRequiredParams(BaseTestClass):
    """Estimator with only optional parameters."""

    def __init__(self, param1=1, param2="test"):
        self.param1 = param1
        self.param2 = param2


class NotBaseClass:
    """Class that doesn't inherit from BaseTestClass."""

    def __init__(self):
        pass


class ClassWithNested(BaseTestClass):
    """Estimator that accepts another estimator as parameter."""

    def __init__(self, estimator, value=5):
        self.estimator = estimator
        self.value = value


class ClassWithOptional(BaseTestClass):
    """Estimator with optional parameters that can be None."""

    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2


class ClassWithInner(BaseTestClass):
    """Estimator that accepts an inner object."""

    def __init__(self, inner=None):
        self.inner = inner


# ============================================================================
# Wrapper Classes
# ============================================================================


class SimpleWrapper(BaseClassWrapper):
    """Concrete wrapper implementation for testing."""

    _estimator_name = "simple"
    _estimator_base_class = BaseTestClass


class MissingNameWrapper(BaseClassWrapper):
    """Wrapper without _estimator_name defined."""

    _estimator_base_class = BaseTestClass


class MissingBaseClassWrapper(BaseClassWrapper):
    """Wrapper without _estimator_base_class defined."""

    _estimator_name = "simple"


class DefaultClassWrapper(BaseClassWrapper):
    """Wrapper with a default estimator class."""

    _estimator_name = "simple"
    _estimator_base_class = BaseTestClass
    _estimator_default_class = NoRequiredParams


# ============================================================================
# Constants
# ============================================================================


REQUIRED_PARAM_TEST_VALUE = 42


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def base_test_class():
    """Fixture providing BaseTestClass."""
    return BaseTestClass


@pytest.fixture
def simple_estimator():
    """Fixture providing SimpleEstimator class."""
    return SimpleEstimator


@pytest.fixture
def no_required_params():
    """Fixture providing NoRequiredParams class."""
    return NoRequiredParams


@pytest.fixture
def not_base_class():
    """Fixture providing NotBaseClass."""
    return NotBaseClass


@pytest.fixture
def simple_wrapper():
    """Fixture providing SimpleWrapper class."""
    return SimpleWrapper


@pytest.fixture
def missing_name_wrapper():
    """Fixture providing MissingNameWrapper class."""
    return MissingNameWrapper


@pytest.fixture
def missing_base_class_wrapper():
    """Fixture providing MissingBaseClassWrapper class."""
    return MissingBaseClassWrapper


@pytest.fixture
def required_param_value():
    """Fixture providing test value for required parameters."""
    return REQUIRED_PARAM_TEST_VALUE
