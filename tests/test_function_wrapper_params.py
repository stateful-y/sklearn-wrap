"""Tests for FunctionWrapper get_params, set_params, clone roundtrip."""

import functools

import pytest
from sklearn.base import clone

from sklearn_wrap.base import FunctionWrapper

from .conftest import (
    DefaultCallableWrapper,
    NoRequiredParams,
    SimpleFunctionWrapper,
    SimpleWrapper,
    kwargs_fn,
    predict_fn,
    simple_fn,
)


class TestGetParams:
    """Tests for FunctionWrapper.get_params."""

    def test_get_params_includes_callable(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0)
        params = wrapper.get_params()
        assert params["fn"] is simple_fn
        assert params["alpha"] == 5.0
        assert params["beta"] == 2.0

    def test_get_params_default_callable(self):
        wrapper = DefaultCallableWrapper()
        params = wrapper.get_params()
        assert params["fn"] is simple_fn
        assert params["alpha"] == 1.0

    def test_get_params_deep_false(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0)
        params = wrapper.get_params(deep=False)
        assert params["fn"] is simple_fn
        assert params["alpha"] == 5.0

    def test_get_params_nested_wrapper(self):
        inner = SimpleFunctionWrapper(fn=simple_fn, alpha=3.0)

        class OuterWrapper(FunctionWrapper):
            _callable_name = "fn"

        outer = OuterWrapper(fn=kwargs_fn, base=1.0, inner=inner)
        params = outer.get_params(deep=True)
        assert params["inner"] is inner
        assert params["inner__alpha"] == 3.0
        assert params["inner__beta"] == 2.0
        # callable key of nested wrapper should NOT appear in flattened params
        assert "inner__fn" not in params

    def test_get_params_nested_base_class_wrapper(self):
        inner = SimpleWrapper(simple=NoRequiredParams, param1=99)

        class OuterWrapper(FunctionWrapper):
            _callable_name = "fn"

        outer = OuterWrapper(fn=kwargs_fn, base=1.0, nested=inner)
        params = outer.get_params(deep=True)
        assert params["nested"] is inner
        assert params["nested__param1"] == 99
        # estimator_name key of nested BaseClassWrapper should NOT appear
        assert "nested__simple" not in params

    def test_get_params_nested_plain_estimator(self):
        from sklearn.tree import DecisionTreeClassifier

        inner = DecisionTreeClassifier(max_depth=3)

        class OuterWrapper(FunctionWrapper):
            _callable_name = "fn"

        outer = OuterWrapper(fn=kwargs_fn, base=1.0, tree=inner)
        params = outer.get_params(deep=True)
        assert params["tree"] is inner
        assert params["tree__max_depth"] == 3


class TestSetParams:
    """Tests for FunctionWrapper.set_params."""

    def test_set_simple_params(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=1.0)
        wrapper.set_params(alpha=10.0)
        assert wrapper._params["alpha"] == 10.0

    def test_set_params_returns_self(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        result = wrapper.set_params(alpha=2.0)
        assert result is wrapper

    def test_set_params_rejects_callable_change(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        with pytest.raises(ValueError, match="Cannot change callable"):
            wrapper.set_params(fn=predict_fn)

    def test_set_params_empty(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        result = wrapper.set_params()
        assert result is wrapper

    def test_set_params_invalid_raises(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        with pytest.raises(ValueError, match="not a valid keyword-only parameter"):
            wrapper.set_params(nonexistent=1.0)

    def test_set_nested_params(self):
        inner = SimpleFunctionWrapper(fn=simple_fn, alpha=1.0)

        class OuterWrapper(FunctionWrapper):
            _callable_name = "fn"

        outer = OuterWrapper(fn=kwargs_fn, base=1.0, inner=inner)
        outer.set_params(inner__alpha=99.0)
        assert inner._params["alpha"] == 99.0

    def test_set_nested_invalid_base_key_raises(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        with pytest.raises(ValueError, match="Invalid parameter"):
            wrapper.set_params(nonexistent__sub=1.0)

    def test_set_nested_no_set_params_raises(self):
        wrapper = SimpleFunctionWrapper(fn=kwargs_fn, base=1.0, obj="plain_string")
        with pytest.raises(AttributeError, match="does not have a set_params"):
            wrapper.set_params(obj__sub=1.0)

    def test_set_params_double_underscore_rejects(self):
        # Double-underscore in set_params is treated as nested param syntax,
        # so it raises because the base key doesn't exist in _params
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        with pytest.raises(ValueError, match="Invalid parameter"):
            wrapper.set_params(bad__param=1.0)


class TestCloneRoundtrip:
    """Tests for sklearn.base.clone compatibility."""

    def test_clone_preserves_params(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0, beta=10.0)
        cloned = clone(wrapper)
        assert cloned.callable_fn is simple_fn
        assert cloned._params["alpha"] == 5.0
        assert cloned._params["beta"] == 10.0

    def test_clone_preserves_callable_identity(self):
        wrapper = SimpleFunctionWrapper(fn=predict_fn, scale=2.0)
        cloned = clone(wrapper)
        assert cloned.callable_fn is predict_fn

    def test_clone_default_callable(self):
        wrapper = DefaultCallableWrapper(alpha=3.0)
        cloned = clone(wrapper)
        assert cloned.callable_fn is simple_fn
        assert cloned._params["alpha"] == 3.0

    def test_clone_is_independent(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=1.0)
        cloned = clone(wrapper)
        cloned.set_params(alpha=99.0)
        assert wrapper._params["alpha"] == 1.0
        assert cloned._params["alpha"] == 99.0

    def test_clone_not_fitted(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn)
        wrapper._fitted = True
        cloned = clone(wrapper)
        assert not cloned.__sklearn_is_fitted__()

    def test_clone_lambda(self):
        fn = lambda X, *, k=1: X * k  # noqa: E731
        wrapper = SimpleFunctionWrapper(fn=fn, k=5)
        cloned = clone(wrapper)
        assert cloned.callable_fn is fn
        assert cloned._params["k"] == 5

    def test_clone_partial(self):
        base_fn = lambda X, y, *, alpha=1.0, beta=2.0: X * alpha + beta  # noqa: E731
        partial_fn = functools.partial(base_fn, beta=10.0)
        wrapper = SimpleFunctionWrapper(fn=partial_fn, alpha=3.0)
        cloned = clone(wrapper)
        # sklearn's clone deep-copies non-estimator values, so partial identity
        # is not preserved, but behavior and params should match
        assert cloned._params["alpha"] == 3.0
        assert cloned._params["beta"] == 10.0
        assert cloned(5, 0) == wrapper(5, 0)


class TestRepr:
    """Tests that repr works without errors (no RecursionError)."""

    def test_repr_simple(self):
        wrapper = SimpleFunctionWrapper(fn=simple_fn, alpha=5.0)
        r = repr(wrapper)
        assert "SimpleFunctionWrapper" in r
        assert "alpha=5.0" in r

    def test_repr_default_callable(self):
        wrapper = DefaultCallableWrapper()
        r = repr(wrapper)
        assert "DefaultCallableWrapper" in r

    def test_repr_lambda(self):
        fn = lambda X, *, k=1: X * k  # noqa: E731
        wrapper = SimpleFunctionWrapper(fn=fn, k=3)
        r = repr(wrapper)
        assert "SimpleFunctionWrapper" in r
