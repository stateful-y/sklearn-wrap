"""Tests for sklearn_wrap._validation module."""

import pytest

from sklearn_wrap._validation import validate_class_params, validate_dotted_path, validate_function_params


class TestValidateDottedPath:
    """Tests for the validate_dotted_path utility."""

    def test_valid_single_segment(self):
        assert validate_dotted_path("sklearn") == ["sklearn"]

    def test_valid_multi_segment(self):
        assert validate_dotted_path("sklearn.linear_model.Ridge") == [
            "sklearn",
            "linear_model",
            "Ridge",
        ]

    def test_min_segments_enforced(self):
        with pytest.raises(ValueError, match="Invalid dotted path"):
            validate_dotted_path("sklearn", min_segments=2)

    def test_min_segments_two_passes(self):
        assert validate_dotted_path("sklearn.Ridge", min_segments=2) == [
            "sklearn",
            "Ridge",
        ]

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid dotted path"):
            validate_dotted_path("")

    def test_non_identifier_rejected(self):
        with pytest.raises(ValueError, match="Invalid dotted path"):
            validate_dotted_path("sklearn.123bad")

    def test_trailing_dot_rejected(self):
        with pytest.raises(ValueError, match="Invalid dotted path"):
            validate_dotted_path("sklearn.")


class TestValidateClassParams:
    """Tests for the validate_class_params utility."""

    def test_all_valid_params(self):
        class Foo:
            def __init__(self, a, b=2):
                pass

        result = validate_class_params(Foo, {"a": 1, "b": 3})
        assert result == {"a": 1, "b": 3}

    def test_fills_defaults_for_missing(self):
        class Foo:
            def __init__(self, a, b=2, c="x"):
                pass

        result = validate_class_params(Foo, {"a": 1})
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == "x"

    def test_required_param_gets_sentinel(self):
        class Foo:
            def __init__(self, required):
                pass

        result = validate_class_params(Foo, {})
        assert result["required"] == "__REQUIRED__"

    def test_invalid_param_raises(self):
        class Foo:
            def __init__(self, a=1):
                pass

        with pytest.raises(ValueError, match="'bad' is not a valid parameter"):
            validate_class_params(Foo, {"bad": 42})

    def test_kwargs_accepts_any_param(self):
        class Foo:
            def __init__(self, a=1, **kwargs):
                pass

        result = validate_class_params(Foo, {"a": 1, "extra": 99})
        assert result["a"] == 1
        assert result["extra"] == 99

    def test_kwargs_still_fills_defaults(self):
        class Foo:
            def __init__(self, a=1, **kwargs):
                pass

        result = validate_class_params(Foo, {"extra": 99})
        assert result["a"] == 1
        assert result["extra"] == 99

    def test_preserves_none_values(self):
        class Foo:
            def __init__(self, a=1):
                pass

        result = validate_class_params(Foo, {"a": None})
        assert result["a"] is None

    def test_empty_params_fills_all_defaults(self):
        class Foo:
            def __init__(self, x=10, y=20):
                pass

        result = validate_class_params(Foo, {})
        assert result == {"x": 10, "y": 20}


class TestValidateFunctionParams:
    """Tests for the validate_function_params utility."""

    def test_uninspectable_callable_raises(self):
        class BadCallable:
            def __call__(self): ...

            __signature__ = property(lambda self: (_ for _ in ()).throw(ValueError("no sig")))

        with pytest.raises(TypeError, match="Cannot inspect signature"):
            validate_function_params(BadCallable(), {})
