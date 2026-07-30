# Configuration Reference

All class attributes, parameter constraints, and configuration functions available when defining wrapper classes or using `EstimatorConfig`.

## Wrapper Class Attributes

### Required

`_estimator_name` : `str`
:   Parameter name for the wrapped class. Used as the key in `get_params()` and as the keyword argument name in the wrapper constructor.
    Example: `"regressor"`, `"classifier"`, `"model"`.

`_estimator_base_class` : `type`
:   Base class the wrapped class must inherit from. Validated at instantiation. Use `object` for no restriction.

### Optional

`_estimator_default_class` : `type | None`
:   Default class to use when the `_estimator_name` keyword is not passed to the constructor. Defaults to `None` (the class argument is required).

`_parameter_constraints` : `dict[str, list]`
:   Dictionary mapping parameter names to lists of allowed types or constraint objects. When defined, parameters are validated automatically by `_fit_context` before instantiation. Defaults to `{}` (no validation).

## Parameter Constraint Types

Constraints use `sklearn.utils._param_validation`. Each parameter maps to a list of allowed constraints, and the value must match at least one.

| Constraint | Import | Description |
|---|---|---|
| `Interval(type, low, high, closed)` | `sklearn.utils._param_validation` | Numeric range. `closed` is one of `"left"`, `"right"`, `"both"`, `"neither"`. |
| `StrOptions(options)` | `sklearn.utils._param_validation` | Set of allowed strings. |
| `type` | built-in | Exact type check (`int`, `str`, `float`, etc.) |
| `None` | built-in | Allow `None` as a value. |
| `callable` | built-in | Any callable object. |
| `{"wrapper_base_class": type}` | sklearn-wrap specific | Validates that the value is a `BaseClassWrapper` whose `_estimator_base_class` matches the given type. |

## Sentinel Values

`REQUIRED_PARAM_VALUE`
:   String sentinel (`"__REQUIRED__"`) used internally for required parameters without defaults. If any parameter still holds this value when `instantiate()` is called, a `ValueError` is raised.

## Configuration Functions

These functions manage the `EstimatorConfig` trusted modules setting.

### `get_config() -> dict[str, Any]`

Return a copy of the current sklearn-wrap configuration. Thread-local overrides take precedence over global defaults.

### `set_config(*, trusted_modules: frozenset[str] | None = None) -> None`

Persistently update the configuration for the current thread.

### `reset_config() -> None`

Reset the configuration to defaults (`trusted_modules = frozenset({"sklearn", "sklearn_wrap"})`).

### `config_context(*, trusted_modules: frozenset[str] | None = None) -> Generator`

Context manager that temporarily overrides the configuration. Reverts on exit:

```python
from sklearn_wrap.config import config_context

with config_context(trusted_modules=frozenset({"sklearn", "xgboost"})):
    estimator = config.build()
# trusted_modules reverts here
```

## EstimatorConfig

`EstimatorConfig` is a Pydantic `BaseModel` for declarative estimator configuration.

### Fields

| Field | Type | Description |
|---|---|---|
| `estimator_class` | `str` | Dotted import path (e.g. `"sklearn.linear_model.Ridge"`). |
| `params` | `dict[str, Any]` | Constructor keyword arguments. Default: `{}`. |

### Methods

| Method | Signature | Description |
|---|---|---|
| `build` | `(trusted_modules=None) -> object` | Import the class, validate params, return an instance. Uses thread config if `trusted_modules` is `None`. |
| `from_estimator` | `(estimator, *, prune_defaults=True) -> EstimatorConfig` | Class method. Capture config from an existing sklearn-compatible estimator. With `prune_defaults=True` (the default), parameters left at their constructor default are omitted at every level of nesting; pass `False` to record the full parameter set. |
| `from_yaml` | `(path) -> EstimatorConfig` | Class method. Load from a YAML file. Supports `!include` tags. |
| `to_yaml` | `(path) -> None` | Save the config to a YAML file. |

### Security

`build()` only resolves classes from trusted modules. The default set is `{"sklearn", "sklearn_wrap"}`. Use `set_config` or `config_context` to add modules like `"xgboost"`. The `UntrustedModuleError` exception is raised for disallowed modules.

## See Also

- [How to Use YAML Configuration](../how-to/yaml-configuration.md): practical usage guide
- [How to Validate Parameters](../how-to/validate-parameters.md): using `_parameter_constraints`
- [API Reference](api.md): auto-generated API docs
