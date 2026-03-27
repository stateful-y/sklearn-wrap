"""Tests for sklearn_wrap.config module."""

import pickle
import textwrap

import pytest
import yaml
from pydantic import ValidationError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_wrap.config import (
    DEFAULT_TRUSTED_MODULES,
    EstimatorConfig,
    UntrustedModuleError,
    _class_to_dotted_path,
    _ClassRef,
    _import_class,
    _load_yaml,
    _resolve_params,
    _resolve_value,
    _serialize_value,
)


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


class TestClassToDottedPath:
    """Tests for _class_to_dotted_path utility."""

    def test_sklearn_class(self):
        path = _class_to_dotted_path(Ridge)
        assert "Ridge" in path
        assert "sklearn" in path

    def test_builtin(self):
        assert _class_to_dotted_path(dict) == "builtins.dict"


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
        result = _serialize_value(Ridge)
        assert result == {"__type__": _class_to_dotted_path(Ridge)}

    def test_serialize_value_dict(self):
        """_serialize_value recursively handles plain dicts."""
        result = _serialize_value({"alpha": 1.0})
        assert result == {"alpha": 1.0}
