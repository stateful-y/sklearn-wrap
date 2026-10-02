"""Tests for sklearn_wrap.config module."""

import concurrent.futures
import importlib
import inspect
import pickle
import sys
import textwrap

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_wrap.base import BaseClassWrapper
from sklearn_wrap.config import (
    DEFAULT_TRUSTED_MODULES,
    EstimatorConfig,
    UntrustedModuleError,
    _class_to_dotted_path,
    _ClassRef,
    _constructor_defaults,
    _equals_default,
    _import_class,
    _load_yaml,
    _prune_defaults,
    _resolve_params,
    _resolve_value,
    _serialize_value,
    _threadlocal,
    config_context,
    get_config,
    reset_config,
    set_config,
)

from .conftest import DefaultClassWrapper, NoRequiredParams

# Configs derived from estimators defined in this file name `tests.*` paths, which
# the default allowlist does not cover.
TEST_TRUSTED_MODULES = frozenset({"sklearn", "sklearn_wrap", "tests"})


class RandomForestWrapper(BaseClassWrapper, RegressorMixin):
    """Wrapper over a regressor, used to exercise pruning on the wrapper type."""

    _estimator_name = "regressor"
    _estimator_base_class = RegressorMixin


class ExtraParamEstimator(BaseEstimator):
    """Estimator whose get_params reports a key its constructor does not declare.

    Breaks the scikit-learn convention on purpose: pruning has no default to
    compare such a key against, so it must be kept.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_params(self, deep=True):
        """Return the declared parameter plus one the constructor never accepts."""
        return {"alpha": self.alpha, "undeclared": "kept"}


class TestImportClass:
    """Tests for _import_class utility."""

    @pytest.mark.parametrize(
        "dotted_path,expected_class",
        [
            pytest.param("sklearn.linear_model.Ridge", Ridge, id="sklearn_class"),
            pytest.param("sklearn.preprocessing.StandardScaler", StandardScaler, id="nested_module"),
        ],
    )
    def test_import_success(self, dotted_path, expected_class):
        cls = _import_class(dotted_path)
        assert cls is expected_class

    def test_untrusted_module_rejected(self):
        with pytest.raises(UntrustedModuleError, match="not in the trusted modules list"):
            _import_class("os.path")

    def test_custom_trusted_modules(self):
        cls = _import_class("builtins.dict", trusted_modules=frozenset({"builtins"}))
        assert cls is dict

    @pytest.mark.parametrize(
        "dotted_path,error_type,match",
        [
            pytest.param("", ImportError, "Invalid dotted path", id="empty_path"),
            pytest.param("sklearn.123bad", ImportError, "Invalid dotted path", id="invalid_identifier"),
            pytest.param(
                "sklearn.linear_model.NonExistentClass", ImportError, "Could not import", id="nonexistent_class"
            ),
            pytest.param("sklearn.nonexistent_module.Foo", ImportError, "Could not import", id="nonexistent_module"),
            pytest.param("sklearn.__version__", ImportError, "not a class", id="non_class_attribute"),
        ],
    )
    def test_import_errors(self, dotted_path, error_type, match):
        with pytest.raises(error_type, match=match):
            _import_class(dotted_path)


@pytest.fixture
def reexporting_package(tmp_path):
    """Build an importable package that re-exports a class from a submodule.

    Stands in for libraries such as LightGBM or XGBoost, which define their
    estimators in a submodule and document them at the top level. Neither is a
    test dependency of this project, so the case is built rather than borrowed.
    """
    name = "_swrap_reexport_pkg"
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("from .inner import Thing\n\n__all__ = ['Thing']\n")
    (package / "inner.py").write_text("class Thing:\n    pass\n")

    sys.path.insert(0, str(tmp_path))
    try:
        yield importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))
        for module in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
            del sys.modules[module]
        importlib.invalidate_caches()


class TestClassToDottedPath:
    """Tests for _class_to_dotted_path utility."""

    def test_sklearn_class(self):
        path = _class_to_dotted_path(Ridge)
        assert "Ridge" in path
        assert "sklearn" in path

    def test_builtin(self):
        assert _class_to_dotted_path(dict) == "builtins.dict"

    def test_private_defining_module_resolves_to_public_parent(self):
        """A class defined in a private module is recorded at its public re-export."""
        assert Ridge.__module__ == "sklearn.linear_model._ridge"
        assert _class_to_dotted_path(Ridge) == "sklearn.linear_model.Ridge"

    def test_already_public_module_is_unchanged(self):
        """A class whose defining module is already the public one is left alone."""
        assert _class_to_dotted_path(Pipeline) == "sklearn.pipeline.Pipeline"

    def test_top_level_reexport_resolves_to_package(self, reexporting_package):
        """A submodule class re-exported by its package is recorded at the top level."""
        assert reexporting_package.Thing.__module__ == "_swrap_reexport_pkg.inner"
        assert _class_to_dotted_path(reexporting_package.Thing) == "_swrap_reexport_pkg.Thing"

    def test_no_reexport_falls_back_to_defining_path(self):
        """A class no ancestor module exposes keeps its defining path."""

        class Unexported:
            pass

        path = _class_to_dotted_path(Unexported)
        assert path == f"{Unexported.__module__}.{Unexported.__qualname__}"

    def test_nested_class_falls_back_to_qualified_name(self):
        """A nested class is not probed for re-export; its qualified name is kept."""

        class Outer:
            class Inner:
                pass

        path = _class_to_dotted_path(Outer.Inner)
        assert path.endswith("Outer.Inner")
        assert path == f"{Outer.Inner.__module__}.{Outer.Inner.__qualname__}"

    def test_class_bound_to_no_module_attribute_falls_back(self):
        """Shortening requires a real re-export, not merely a public-looking ancestor.

        This class claims a private defining module but is bound nowhere, so the
        walk finds no ancestor exposing it and the private path is kept.
        """
        orphan = type("Orphan", (), {"__module__": "sklearn.linear_model._ridge"})
        assert orphan.__qualname__ == orphan.__name__
        assert not hasattr(sys.modules["sklearn.linear_model"], "Orphan")
        assert _class_to_dotted_path(orphan) == "sklearn.linear_model._ridge.Orphan"

    def test_resolution_imports_nothing(self):
        """Resolving a path must not import modules as a side effect."""
        before = set(sys.modules)
        _class_to_dotted_path(Ridge)
        _class_to_dotted_path(Pipeline)
        _class_to_dotted_path(dict)
        assert set(sys.modules) == before


class TestEstimatorConfigValidation:
    """Tests for EstimatorConfig field validation."""

    def test_valid_config(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0},
        )
        assert config.estimator_class == "sklearn.linear_model.Ridge"
        assert config.params == {"alpha": 1.0}

    def test_default_empty_params(self):
        config = EstimatorConfig(estimator_class="sklearn.linear_model.Ridge")
        assert config.params == {}

    def test_rejects_single_segment_path(self):
        with pytest.raises(ValueError, match="valid dotted import path"):
            EstimatorConfig(estimator_class="Ridge")

    def test_rejects_invalid_identifier(self):
        with pytest.raises(ValueError, match="valid dotted import path"):
            EstimatorConfig(estimator_class="sklearn.123.Ridge")

    def test_nested_estimator_config_auto_conversion(self):
        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={
                "steps": [
                    [
                        "scaler",
                        {
                            "estimator_class": "sklearn.preprocessing.StandardScaler",
                        },
                    ],
                ],
            },
        )
        step = config.params["steps"][0]
        assert step[0] == "scaler"
        assert isinstance(step[1], EstimatorConfig)
        assert step[1].estimator_class == "sklearn.preprocessing.StandardScaler"

    def test_class_ref_auto_conversion(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"some_class": {"__type__": "sklearn.preprocessing.StandardScaler"}},
        )
        assert isinstance(config.params["some_class"], _ClassRef)
        assert config.params["some_class"].type_path == "sklearn.preprocessing.StandardScaler"

    def test_plain_dict_preserved(self):
        """Dict without estimator_class or __type__ stays as dict."""
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"meta": {"key": "value", "n": 42}},
        )
        assert config.params["meta"] == {"key": "value", "n": 42}

    def test_model_validate_non_dict_raises(self):
        """_convert_nested handles non-dict input before pydantic rejects it."""
        with pytest.raises(ValidationError):
            EstimatorConfig.model_validate("not a dict")


class TestEstimatorConfigBuild:
    """Tests for building estimators from config."""

    def test_build_simple_estimator(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 2.0, "fit_intercept": False},
        )
        est = config.build()
        assert est.__class__.__name__ == "Ridge"
        assert est.alpha == 2.0
        assert est.fit_intercept is False

    def test_build_default_params(self):
        config = EstimatorConfig(estimator_class="sklearn.linear_model.Ridge")
        est = config.build()
        assert est.alpha == 1.0  # sklearn default

    def test_build_pipeline(self):
        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={
                "steps": [
                    ["scaler", {"estimator_class": "sklearn.preprocessing.StandardScaler"}],
                    ["ridge", {"estimator_class": "sklearn.linear_model.Ridge", "params": {"alpha": 0.5}}],
                ],
            },
        )
        pipe = config.build()
        assert pipe.__class__.__name__ == "Pipeline"
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "scaler"
        assert pipe.steps[1][0] == "ridge"
        assert pipe.steps[1][1].alpha == 0.5

    def test_build_with_class_ref(self):
        """__type__ dicts resolve to actual classes during build."""
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"some_class": {"__type__": "sklearn.preprocessing.StandardScaler"}},
        )
        # Ridge doesn't actually accept some_class, but we test resolution
        # by checking the resolved params dict
        resolved = _resolve_params(config.params, trusted_modules=DEFAULT_TRUSTED_MODULES)
        assert resolved["some_class"] is StandardScaler

    def test_build_untrusted_module_rejected(self):
        config = EstimatorConfig(
            estimator_class="os.path.join",
            params={},
        )
        with pytest.raises(UntrustedModuleError):
            config.build()

    def test_build_custom_trusted_modules(self):
        config = EstimatorConfig(
            estimator_class="builtins.dict",
        )
        result = config.build(trusted_modules=frozenset({"builtins"}))
        assert isinstance(result, dict)


class TestFromEstimator:
    """Tests for creating configs from existing estimators."""

    def test_simple_estimator(self):
        est = Ridge(alpha=3.0)
        config = EstimatorConfig.from_estimator(est)
        assert "Ridge" in config.estimator_class
        assert config.params["alpha"] == 3.0

    def test_pipeline(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
        config = EstimatorConfig.from_estimator(pipe)
        assert "Pipeline" in config.estimator_class
        # steps are serialized as list of [name, config_dict] pairs
        steps = config.params["steps"]
        assert len(steps) == 2
        assert steps[0][0] == "scaler"
        assert steps[1][0] == "ridge"

    def test_roundtrip_simple(self):
        """from_estimator -> build reproduces equivalent params."""
        original = Ridge(alpha=5.0, fit_intercept=False)
        config = EstimatorConfig.from_estimator(original)
        rebuilt = config.build()
        assert rebuilt.alpha == original.alpha
        assert rebuilt.fit_intercept == original.fit_intercept


class TestYamlIO:
    """Tests for YAML serialization and deserialization."""

    def test_to_yaml_and_from_yaml_roundtrip(self, tmp_path):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 2.0, "fit_intercept": True},
        )
        path = tmp_path / "config.yaml"
        config.to_yaml(path)

        loaded = EstimatorConfig.from_yaml(path)
        assert loaded.estimator_class == config.estimator_class
        assert loaded.params == config.params

    def test_yaml_with_anchors_and_merge_keys(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            _defaults: &defaults
              fit_intercept: true

            estimator_class: sklearn.linear_model.Ridge
            params:
              <<: *defaults
              alpha: 1.5
        """)
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        config = EstimatorConfig.from_yaml(path)
        assert config.params["alpha"] == 1.5
        assert config.params["fit_intercept"] is True

    def test_yaml_anchor_override(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            _defaults: &defaults
              alpha: 1.0
              fit_intercept: true

            estimator_class: sklearn.linear_model.Ridge
            params:
              <<: *defaults
              alpha: 99.0
        """)
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        config = EstimatorConfig.from_yaml(path)
        assert config.params["alpha"] == 99.0  # override wins

    def test_yaml_include(self, tmp_path):
        scaler_yaml = textwrap.dedent("""\
            estimator_class: sklearn.preprocessing.StandardScaler
            params: {}
        """)
        (tmp_path / "scaler.yaml").write_text(scaler_yaml)

        main_yaml = textwrap.dedent("""\
            estimator_class: sklearn.pipeline.Pipeline
            params:
              steps:
                - - scaler
                  - !include scaler.yaml
                - - ridge
                  - estimator_class: sklearn.linear_model.Ridge
                    params:
                      alpha: 0.1
        """)
        (tmp_path / "main.yaml").write_text(main_yaml)

        config = EstimatorConfig.from_yaml(tmp_path / "main.yaml")
        steps = config.params["steps"]
        assert steps[0][0] == "scaler"
        assert isinstance(steps[0][1], EstimatorConfig)
        assert steps[0][1].estimator_class == "sklearn.preprocessing.StandardScaler"

    def test_yaml_include_nested_dirs(self, tmp_path):
        sub_dir = tmp_path / "models"
        sub_dir.mkdir()
        (sub_dir / "ridge.yaml").write_text(
            textwrap.dedent("""\
                estimator_class: sklearn.linear_model.Ridge
                params:
                  alpha: 0.5
            """)
        )
        main_yaml = textwrap.dedent("""\
            estimator_class: sklearn.pipeline.Pipeline
            params:
              steps:
                - - ridge
                  - !include models/ridge.yaml
        """)
        (tmp_path / "main.yaml").write_text(main_yaml)

        config = EstimatorConfig.from_yaml(tmp_path / "main.yaml")
        step = config.params["steps"][0]
        assert isinstance(step[1], EstimatorConfig)
        assert step[1].params["alpha"] == 0.5

    def test_yaml_circular_include_detected(self, tmp_path):
        (tmp_path / "a.yaml").write_text("data: !include b.yaml\n")
        (tmp_path / "b.yaml").write_text("data: !include a.yaml\n")

        with pytest.raises(yaml.YAMLError, match="Circular include"):
            _load_yaml(tmp_path / "a.yaml")


class TestFullRoundtrip:
    """End-to-end roundtrip tests."""

    def test_ridge_roundtrip_via_yaml(self, tmp_path):
        original = Ridge(alpha=7.0, fit_intercept=False)
        config = EstimatorConfig.from_estimator(original)

        path = tmp_path / "ridge.yaml"
        config.to_yaml(path)

        loaded = EstimatorConfig.from_yaml(path)
        rebuilt = loaded.build()

        assert rebuilt.alpha == 7.0
        assert rebuilt.fit_intercept is False

    def test_pipeline_roundtrip_via_yaml(self, tmp_path):
        original = Pipeline([("scaler", StandardScaler(with_mean=False)), ("ridge", Ridge(alpha=0.3))])
        config = EstimatorConfig.from_estimator(original)

        path = tmp_path / "pipeline.yaml"
        config.to_yaml(path)

        loaded = EstimatorConfig.from_yaml(path)
        rebuilt = loaded.build()

        assert rebuilt.__class__.__name__ == "Pipeline"
        assert len(rebuilt.steps) == 2
        assert rebuilt.steps[0][1].with_mean is False
        assert rebuilt.steps[1][1].alpha == 0.3


class TestSecurity:
    """Tests for security restrictions on class resolution."""

    def test_default_trusted_modules(self):
        assert "sklearn" in DEFAULT_TRUSTED_MODULES
        assert "sklearn_wrap" in DEFAULT_TRUSTED_MODULES

    @pytest.mark.parametrize(
        "dotted_path",
        [
            pytest.param("subprocess.Popen", id="subprocess"),
            pytest.param("os.system", id="os"),
            pytest.param("builtins.eval", id="builtins"),
            pytest.param("builtins.__import__", id="dunder_import"),
        ],
    )
    def test_untrusted_module_rejected(self, dotted_path):
        """Verify various untrusted modules are rejected by the allowlist."""
        with pytest.raises(UntrustedModuleError):
            _import_class(dotted_path)

    def test_build_rejects_untrusted_nested(self):
        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={
                "steps": [
                    ["evil", {"estimator_class": "subprocess.Popen"}],
                ],
            },
        )
        with pytest.raises(UntrustedModuleError):
            config.build()


class TestEdgeCases:
    """Edge case tests."""

    def test_config_is_picklable(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0},
        )
        roundtripped = pickle.loads(pickle.dumps(config))
        assert roundtripped.estimator_class == config.estimator_class
        assert roundtripped.params == config.params

    def test_model_dump_serializable(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0},
        )
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["estimator_class"] == "sklearn.linear_model.Ridge"

    def test_empty_params(self):
        config = EstimatorConfig(estimator_class="sklearn.linear_model.Ridge")
        est = config.build()
        assert est.__class__.__name__ == "Ridge"


class TestGlobalConfig:
    """Tests for set_config / get_config / config_context / reset_config."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        """Ensure clean config state for every test."""
        reset_config()
        yield
        reset_config()

    def test_get_config_returns_defaults(self):
        cfg = get_config()
        assert cfg["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_set_config_updates_trusted_modules(self):
        new_modules = frozenset({"sklearn", "xgboost"})
        set_config(trusted_modules=new_modules)
        assert get_config()["trusted_modules"] == new_modules

    def test_set_config_none_is_noop(self):
        set_config(trusted_modules=None)
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_set_config_initializes_threadlocal(self):
        """set_config works even when threadlocal has no config attribute."""
        if hasattr(_threadlocal, "config"):
            del _threadlocal.config
        set_config(trusted_modules=frozenset({"fresh"}))
        assert get_config()["trusted_modules"] == frozenset({"fresh"})

    def test_reset_config_restores_defaults(self):
        set_config(trusted_modules=frozenset({"custom"}))
        reset_config()
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_config_context_temporary(self):
        with config_context(trusted_modules=frozenset({"sklearn", "xgboost"})):
            assert "xgboost" in get_config()["trusted_modules"]
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_config_context_nested(self):
        with config_context(trusted_modules=frozenset({"a"})):
            assert get_config()["trusted_modules"] == frozenset({"a"})
            with config_context(trusted_modules=frozenset({"b"})):
                assert get_config()["trusted_modules"] == frozenset({"b"})
            assert get_config()["trusted_modules"] == frozenset({"a"})
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_config_context_restores_on_exception(self):
        with pytest.raises(RuntimeError), config_context(trusted_modules=frozenset({"temp"})):
            assert get_config()["trusted_modules"] == frozenset({"temp"})
            raise RuntimeError("boom")
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_config_context_none_is_noop(self):
        with config_context(trusted_modules=None):
            assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_thread_isolation(self):
        """Changes in one thread do not affect another."""
        barrier = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def worker():
            set_config(trusted_modules=frozenset({"thread_pkg"}))
            return get_config()["trusted_modules"]

        future = barrier.submit(worker)
        worker_result = future.result()
        barrier.shutdown()

        assert worker_result == frozenset({"thread_pkg"})
        # Main thread unaffected
        assert get_config()["trusted_modules"] == DEFAULT_TRUSTED_MODULES

    def test_get_config_returns_copy(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_build_uses_global_config(self):
        set_config(trusted_modules=frozenset({"sklearn", "builtins"}))
        config = EstimatorConfig(estimator_class="builtins.dict")
        result = config.build()
        assert isinstance(result, dict)

    def test_build_explicit_overrides_global(self):
        set_config(trusted_modules=frozenset({"sklearn"}))
        config = EstimatorConfig(estimator_class="builtins.dict")
        result = config.build(trusted_modules=frozenset({"builtins"}))
        assert isinstance(result, dict)


class TestBuildParamValidation:
    """Tests for parameter validation at build() time."""

    def test_build_validates_params(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0, "nonexistent_param": 42},
        )
        with pytest.raises(ValueError, match="'nonexistent_param' is not a valid parameter"):
            config.build()

    def test_build_validate_params_false_skips(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0},
        )
        est = config.build(validate_params=False)
        assert est.alpha == 1.0

    def test_build_pipeline_with_kwargs(self):
        """Pipeline accepts **kwargs so no false positives from validation."""
        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={
                "steps": [
                    ["ridge", {"estimator_class": "sklearn.linear_model.Ridge"}],
                ],
            },
        )
        pipe = config.build()
        assert pipe.__class__.__name__ == "Pipeline"

    def test_build_nested_config_validates(self):
        """Nested EstimatorConfig objects are resolved before validation."""
        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={
                "steps": [
                    ["scaler", {"estimator_class": "sklearn.preprocessing.StandardScaler"}],
                    [
                        "ridge",
                        {
                            "estimator_class": "sklearn.linear_model.Ridge",
                            "params": {"alpha": 0.5},
                        },
                    ],
                ],
            },
        )
        pipe = config.build()
        assert len(pipe.steps) == 2
        assert pipe.steps[1][1].alpha == 0.5

    def test_build_valid_params_pass(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 2.0, "fit_intercept": False},
        )
        est = config.build()
        assert est.alpha == 2.0
        assert est.fit_intercept is False

    def test_none_param_value(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model.Ridge",
            params={"alpha": 1.0, "solver": None},
        )
        est = config.build()
        assert est.solver is None

    def test_class_ref_invalid_type_path(self):
        """_ClassRef rejects type_path with invalid identifier segments."""
        with pytest.raises(ValueError, match="Invalid dotted path"):
            _ClassRef(type_path="sklearn.123bad")

    def test_resolve_value_plain_dict(self):
        """_resolve_value passes plain dicts through recursively."""
        result = _resolve_value({"key": "val"}, trusted_modules=DEFAULT_TRUSTED_MODULES)
        assert result == {"key": "val"}

    def test_serialize_value_class(self):
        """_serialize_value converts a class to a __type__ dict."""
        result = _serialize_value(Ridge, prune_defaults=True)
        assert result == {"__type__": _class_to_dotted_path(Ridge)}

    def test_serialize_value_dict(self):
        """_serialize_value recursively handles plain dicts."""
        result = _serialize_value({"alpha": 1.0}, prune_defaults=True)
        assert result == {"alpha": 1.0}


class TestEqualsDefault:
    """Tests for the conservative default comparison."""

    def test_identity_matches(self):
        """The same object is always the default."""
        sentinel = object()
        assert _equals_default(sentinel, sentinel) is True

    def test_equal_values_match(self):
        """Distinct but equal values of the same type match."""
        assert _equals_default(1.0, 1.0) is True
        assert _equals_default("auto", "auto") is True

    def test_differing_values_do_not_match(self):
        assert _equals_default(2.0, 1.0) is False

    @pytest.mark.parametrize(
        "value,default",
        [
            pytest.param(True, 1, id="bool_vs_int"),
            pytest.param(0, 0.0, id="int_vs_float"),
            pytest.param(0, False, id="int_vs_bool"),
        ],
    )
    def test_type_mismatch_is_kept(self, value, default):
        """Values equal under `==` but of a different type are not the default."""
        assert value == default  # the trap this guards against
        assert _equals_default(value, default) is False

    def test_array_comparison_is_kept(self):
        """A comparison yielding an array rather than a bool is not a match."""
        assert _equals_default(np.array([1, 2]), np.array([1, 2])) is False

    def test_raising_comparison_is_kept(self):
        """A comparison that raises is not a match, and the error does not escape."""

        class Explosive:
            __hash__ = object.__hash__

            def __eq__(self, other):
                raise RuntimeError("nope")

        assert _equals_default(Explosive(), Explosive()) is False


class TestConstructorDefaults:
    """Tests for where defaults are read from."""

    def test_plain_estimator_uses_own_signature(self):
        defaults = _constructor_defaults(Ridge(alpha=3.0))
        assert defaults["fit_intercept"] is True
        assert "alpha" in defaults

    def test_wrapper_uses_wrapped_class_signature(self):
        """A wrapper's own signature is `**params`, so defaults come from the wrapped class."""
        own_params = inspect.signature(RandomForestWrapper.__init__).parameters
        assert [p.kind for name, p in own_params.items() if name != "self"] == [inspect.Parameter.VAR_KEYWORD]

        defaults = _constructor_defaults(RandomForestWrapper(regressor=RandomForestRegressor, n_estimators=50))
        assert defaults["n_estimators"] == 100
        assert defaults["bootstrap"] is True

    def test_wrapper_without_default_class_keeps_estimator_class(self):
        """With no `_estimator_default_class`, the class entry has no default to match."""
        wrapper = RandomForestWrapper(regressor=RandomForestRegressor)
        assert "regressor" not in _constructor_defaults(wrapper)

    def test_wrapper_with_default_class_exposes_it(self):
        """A declared default class makes the class entry prunable."""
        wrapper = DefaultClassWrapper()
        assert _constructor_defaults(wrapper)["simple"] is NoRequiredParams

    def test_uninspectable_signature_yields_no_defaults(self, mocker):
        """An estimator whose constructor resists inspection simply prunes nothing.

        Some C-extension estimators have no introspectable signature. Nothing can
        be compared against, so every parameter is kept rather than guessed at.
        """
        mocker.patch("sklearn_wrap.config.inspect.signature", side_effect=ValueError("no signature"))
        assert _constructor_defaults(Ridge(alpha=3.0)) == {}


class TestPruneDefaults:
    """Tests for omitting parameters left at their constructor default."""

    def test_explicit_params_survive(self):
        config = EstimatorConfig.from_estimator(Ridge(alpha=3.0))
        assert config.params["alpha"] == 3.0

    def test_default_params_are_omitted(self):
        config = EstimatorConfig.from_estimator(Ridge(alpha=3.0))
        assert "fit_intercept" not in config.params
        assert config.params == {"alpha": 3.0}

    def test_pruning_can_be_disabled(self):
        est = Ridge(alpha=3.0)
        config = EstimatorConfig.from_estimator(est, prune_defaults=False)
        assert set(config.params) == set(est.get_params(deep=False))
        assert config.params["fit_intercept"] is True

    def test_pruned_config_rebuilds_equivalent_estimator(self):
        original = Ridge(alpha=5.0, fit_intercept=False)
        rebuilt = EstimatorConfig.from_estimator(original).build()
        assert rebuilt.alpha == 5.0
        assert rebuilt.fit_intercept is False

    def test_key_absent_from_signature_is_kept(self):
        """A key `get_params` reports but `__init__` never declares has no default."""
        config = EstimatorConfig.from_estimator(ExtraParamEstimator(alpha=1.0))
        assert config.params["undeclared"] == "kept"
        assert "alpha" not in config.params

    def test_prune_defaults_helper_keeps_unknown_keys(self):
        params = {"alpha": 1.0, "unknown": 7}
        assert _prune_defaults(Ridge(), params) == {"unknown": 7}


class TestNestedPruning:
    """Tests that pruning reaches every level of nesting."""

    def test_nested_step_records_only_what_was_set(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
        config = EstimatorConfig.from_estimator(pipe)
        ridge_step = config.params["steps"][1][1]
        assert ridge_step.params == {"alpha": 0.5}

    def test_fully_default_nested_step_has_empty_params(self):
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
        config = EstimatorConfig.from_estimator(pipe)
        scaler_step = config.params["steps"][0][1]
        assert scaler_step.params == {}

    def test_nested_pruning_is_disabled_with_the_flag(self):
        """The flag reaches nested estimators, not just the outer one."""
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
        config = EstimatorConfig.from_estimator(pipe, prune_defaults=False)
        scaler_step = config.params["steps"][0][1]
        assert scaler_step.params != {}

    def test_nested_pruned_pipeline_rebuilds(self, tmp_path):
        original = Pipeline([("scaler", StandardScaler(with_mean=False)), ("ridge", Ridge(alpha=0.3))])
        path = tmp_path / "pipeline.yaml"
        EstimatorConfig.from_estimator(original).to_yaml(path)

        rebuilt = EstimatorConfig.from_yaml(path).build()
        assert len(rebuilt.steps) == 2
        assert rebuilt.steps[0][1].with_mean is False
        assert rebuilt.steps[1][1].alpha == 0.3


class TestWrapperPruning:
    """Tests that pruning works on BaseClassWrapper, whose defaults are eagerly filled."""

    def test_wrapper_defaults_are_omitted(self):
        wrapper = RandomForestWrapper(regressor=RandomForestRegressor, n_estimators=50)
        config = EstimatorConfig.from_estimator(wrapper)
        assert config.params["n_estimators"] == 50
        assert "criterion" not in config.params
        assert "bootstrap" not in config.params

    def test_wrapper_pruning_is_a_real_reduction(self):
        """The wrapper is the type the naive signature rule would have missed entirely."""
        wrapper = RandomForestWrapper(regressor=RandomForestRegressor, n_estimators=50)
        pruned = EstimatorConfig.from_estimator(wrapper)
        full = EstimatorConfig.from_estimator(wrapper, prune_defaults=False)
        assert len(full.params) == len(wrapper.get_params(deep=False))
        assert len(pruned.params) < len(full.params)

    def test_pruned_wrapper_rebuilds_equivalent_wrapper(self):
        wrapper = RandomForestWrapper(regressor=RandomForestRegressor, n_estimators=50)
        config = EstimatorConfig.from_estimator(wrapper)
        rebuilt = config.build(trusted_modules=TEST_TRUSTED_MODULES)
        assert rebuilt.estimator_class is RandomForestRegressor
        assert rebuilt.params["n_estimators"] == 50

    def test_default_estimator_class_is_omitted_and_restored(self):
        """A wrapper using its declared default class need not record the class."""
        wrapper = DefaultClassWrapper()
        config = EstimatorConfig.from_estimator(wrapper)
        assert "simple" not in config.params

        rebuilt = config.build(trusted_modules=TEST_TRUSTED_MODULES)
        assert rebuilt.estimator_class is NoRequiredParams


class TestPrunedRoundtrip:
    """End-to-end round trips with pruning enabled."""

    def test_pruned_yaml_is_shorter_and_still_builds(self, tmp_path):
        original = Ridge(alpha=7.0, fit_intercept=False)

        pruned_path = tmp_path / "pruned.yaml"
        full_path = tmp_path / "full.yaml"
        EstimatorConfig.from_estimator(original).to_yaml(pruned_path)
        EstimatorConfig.from_estimator(original, prune_defaults=False).to_yaml(full_path)

        pruned_lines = len(pruned_path.read_text().splitlines())
        full_lines = len(full_path.read_text().splitlines())
        assert pruned_lines < full_lines

        rebuilt = EstimatorConfig.from_yaml(pruned_path).build()
        assert rebuilt.alpha == 7.0
        assert rebuilt.fit_intercept is False

    def test_roundtrip_is_stable(self):
        """A second capture of a rebuilt estimator returns exactly the first config."""
        first = EstimatorConfig.from_estimator(Ridge(alpha=5.0, fit_intercept=False))
        second = EstimatorConfig.from_estimator(first.build())
        assert second.model_dump() == first.model_dump()

    def test_nested_roundtrip_is_stable(self):
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("ridge", Ridge(alpha=0.3))])
        first = EstimatorConfig.from_estimator(pipe)
        second = EstimatorConfig.from_estimator(first.build())
        assert second.model_dump() == first.model_dump()


class TestBackwardCompatibility:
    """Configs written before public paths and pruning must keep building."""

    def test_private_defining_path_still_builds(self):
        config = EstimatorConfig(
            estimator_class="sklearn.linear_model._ridge.Ridge",
            params={"alpha": 2.0},
        )
        assert config.build().alpha == 2.0

    def test_exhaustive_config_still_builds(self):
        """A config listing every constructor parameter remains valid input."""
        exhaustive = Ridge(alpha=4.0).get_params(deep=False)
        config = EstimatorConfig(estimator_class="sklearn.linear_model.Ridge", params=exhaustive)
        rebuilt = config.build()
        assert rebuilt.alpha == 4.0
        assert rebuilt.fit_intercept is True

    def test_class_reference_records_public_path_and_resolves(self):
        """A `__type__` marker uses the public path and still imports."""
        serialized = _serialize_value(Ridge, prune_defaults=True)
        assert serialized == {"__type__": "sklearn.linear_model.Ridge"}

        config = EstimatorConfig(
            estimator_class="sklearn.pipeline.Pipeline",
            params={"steps": [], "memory": serialized},
        )
        assert _resolve_params(config.params, trusted_modules=DEFAULT_TRUSTED_MODULES)["memory"] is Ridge
