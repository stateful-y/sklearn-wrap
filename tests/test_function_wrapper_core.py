"""Tests for FunctionWrapper core functionality: init, validation, __call__, fitted checks."""

import pytest

from sklearn_wrap.base import REQUIRED_PARAM_VALUE, FunctionWrapper

from .conftest import (
    DefaultCallableWrapper,
    SimpleFunctionWrapper,
    kwargs_fn,
    no_config_fn,
    predict_fn,
    required_param_fn,
    simple_fn,
)


class TestInit:
    """Tests for FunctionWrapper.__init__."""

    def test_init_with_callable_and_params(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0)
        assert wrapper.callable_fn is simple_fn
        assert wrapper._params["alpha"] == 5.0
        assert wrapper._params["beta"] == 2.0  # default filled

    def test_init_with_default_callable(self):
        wrapper = DefaultCallableWrapper()
        assert wrapper.callable_fn is simple_fn
        assert wrapper._params["alpha"] == 1.0
        assert wrapper._params["beta"] == 2.0

    def test_init_override_default_callable(self):
        wrapper = DefaultCallableWrapper(fn=predict_fn, scale=3.0)
        assert wrapper.callable_fn is predict_fn
        assert wrapper._params["scale"] == 3.0

    def test_init_missing_callable_raises(self):
        with pytest.raises(TypeError, match="missing required keyword argument"):
            SimpleFunctionWrapper(alpha=1.0)

    def test_init_no_callable_name_raises(self):
        class BadWrapper(FunctionWrapper):
            pass

        with pytest.raises(ValueError, match="`_callable_name`"):
            BadWrapper(fn=simple_fn)

    def test_init_not_callable_raises(self):
        with pytest.raises(TypeError, match="not callable"):
            SimpleFunctionWrapper(fn="not_a_function")

    def test_init_required_param_gets_sentinel(self):
        wrapper = SimpleFunctionWrapper(fn=required_param_fn)
        assert wrapper._params["gamma"] == REQUIRED_PARAM_VALUE

    def test_init_no_config_params(self):
        wrapper = SimpleFunctionWrapper(fn=no_config_fn)
        assert wrapper._params == {}

    def test_init_kwargs_fn_accepts_extra_params(self):
        wrapper = SimpleFunctionWrapper(fn=kwargs_fn, base=2.0, extra_param=10.0)
        assert wrapper._params["base"] == 2.0
        assert wrapper._params["extra_param"] == 10.0

    def test_init_invalid_param_raises(self):
        with pytest.raises(ValueError, match="not a valid keyword-only parameter"):
            SimpleFunctionWrapper(fn=simple_fn, nonexistent=1.0)

    def test_init_double_underscore_param_raises(self):
        with pytest.raises(ValueError, match="cannot contain '__'"):
            SimpleFunctionWrapper(fn=kwargs_fn, bad__param=1.0)

    def test_init_lambda(self):
        fn = lambda X, *, k=1: X * k  # noqa: E731
        wrapper = SimpleFunctionWrapper(fn=fn, k=3)
        assert wrapper._params["k"] == 3

    def test_init_class_as_callable(self):
        class MyCallable:
            def __init__(self, *, value=10):
                self.value = value

        wrapper = SimpleFunctionWrapper(fn=MyCallable, value=42)
        assert wrapper._params["value"] == 42


class TestValidation:
    """Tests for validation methods."""

    def test_validate_estimator_fn_not_callable(self):
        with pytest.raises(TypeError, match="not callable"):
            SimpleFunctionWrapper(fn=42)

    def test_validate_params_called_by_instantiate(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0)
        wrapper.instantiate()  # should not raise

    def test_instantiate_required_sentinel_raises(self):
        wrapper = SimpleFunctionWrapper(fn=required_param_fn)
        with pytest.raises(ValueError, match="requires parameter 'gamma'"):
            wrapper.instantiate()


class TestCall:
    """Tests for FunctionWrapper.__call__."""

    def test_call_with_defaults(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        result = wrapper(10, 0)  # X=10, y=0 (data args)
        assert result == 10 * 1.0 + 2.0  # alpha=1.0, beta=2.0

    def test_call_with_custom_params(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=3.0, beta=0.0)
        result = wrapper(5, 0)
        assert result == 5 * 3.0 + 0.0

    def test_call_required_param_raises(self):
        wrapper = SimpleFunctionWrapper(fn=required_param_fn)
        with pytest.raises(ValueError, match="requires parameter"):
            wrapper(10)

    def test_call_no_config_params(self):
        wrapper = SimpleFunctionWrapper(fn=no_config_fn)
        result = wrapper(3, 7)
        assert result == 10

    def test_call_kwargs_fn(self):
        wrapper = SimpleFunctionWrapper(fn=kwargs_fn, base=2.0, bonus=5.0)
        result = wrapper(10)  # 10 * 2.0 + 5.0
        assert result == 25.0


class TestFitted:
    """Tests for __sklearn_is_fitted__."""

    def test_not_fitted_initially(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        assert not wrapper.__sklearn_is_fitted__()

    def test_fitted_after_setting_flag(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        wrapper._fitted = True
        assert wrapper.__sklearn_is_fitted__()

    def test_fitted_with_fitted_attribute(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        wrapper.some_fitted_attr_ = 42
        assert wrapper.__sklearn_is_fitted__()

    def test_callable_fn_not_counted_as_fitted(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        # callable_fn ends with _ but should be excluded
        assert not wrapper.__sklearn_is_fitted__()


class TestInitSubclass:
    """Tests for __init_subclass__ auto-setting _required_parameters."""

    def test_required_parameters_no_default(self):
        assert SimpleFunctionWrapper._required_parameters == ["fn"]

    def test_required_parameters_with_default(self):
        assert DefaultCallableWrapper._required_parameters == []
