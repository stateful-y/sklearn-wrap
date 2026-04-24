"""Shared pytest fixtures and helper classes for sklearn-wrap tests."""

import numpy as np
import pytest

from sklearn_wrap.base import BaseClassWrapper, FunctionWrapper


class BaseTestClass:
    """Base class for test estimators."""

    pass


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


REQUIRED_PARAM_TEST_VALUE = 42


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


# ---------------------------------------------------------------------------
# FunctionWrapper test helpers
# ---------------------------------------------------------------------------


def simple_fn(X, y, *, alpha=1.0, beta=2.0):
    """Function with keyword-only config params."""
    return X * alpha + beta


def required_param_fn(X, *, gamma):
    """Function with a required keyword-only param (no default)."""
    return X * gamma


def kwargs_fn(X, *, base=1.0, **extra):
    """Function accepting arbitrary keyword-only params via **kwargs."""
    return X * base + sum(extra.values())


def no_config_fn(X, y):
    """Function with no keyword-only params (data-only)."""
    return X + y


def predict_fn(X, *, scale=1.0, offset=0.0):
    """A predict-style function for integration tests."""
    return X.sum(axis=1) * scale + offset


class SimpleFunctionWrapper(FunctionWrapper):
    """Concrete FunctionWrapper subclass for testing."""

    _callable_name = "fn"


class DefaultCallableWrapper(FunctionWrapper):
    """FunctionWrapper with a default callable."""

    _callable_name = "fn"
    _callable_default = simple_fn


class FitPredictFunctionWrapper(FunctionWrapper):
    """FunctionWrapper subclass with fit/predict for estimator tests."""

    _callable_name = "fn"

    def fit(self, X, y=None):
        self.fitted_ = True
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.callable_fn(X, **self._params)


@pytest.fixture
def simple_function_wrapper():
    """Fixture providing SimpleFunctionWrapper class."""
    return SimpleFunctionWrapper


@pytest.fixture
def default_callable_wrapper():
    """Fixture providing DefaultCallableWrapper class."""
    return DefaultCallableWrapper
