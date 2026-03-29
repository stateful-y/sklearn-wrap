# How to Use Advanced YAML Patterns

This guide shows you how to use YAML anchors for shared defaults, compose configs across multiple files with `!include`, and manage trusted modules for third-party packages.

!!! tip "Interactive notebook available"

    Try this guide as an interactive notebook:
    [How to Configure Estimators with YAML](/examples/yaml_config/)

## Prerequisites

- Familiarity with the core YAML workflow ([Use YAML Configuration](yaml-configuration.md))
- `sklearn-wrap[config]` installed

## Use YAML Anchors for Shared Defaults

Native YAML anchors let you define shared parameter blocks without custom syntax:

```yaml
_defaults: &defaults
  fit_intercept: true
  solver: auto

estimator_class: sklearn.linear_model.Ridge
params:
  <<: *defaults
  alpha: 0.5
```

The `&defaults` anchor defines a reusable block. The `<<: *defaults` merge key injects those values into `params`. Individual keys like `alpha` override or extend the defaults.

If multiple estimators in a pipeline share common parameters, define the anchor once and reference it in each:

```yaml
_shared: &shared
  fit_intercept: true
  max_iter: 1000
```

## Compose Configs with `!include`

Split complex pipelines across files. Paths resolve relative to the including file:

```yaml
# pipeline.yaml
estimator_class: sklearn.pipeline.Pipeline
params:
  steps:
    - - scaler
      - !include preprocessing.yaml
    - - ridge
      - !include model.yaml
```

```yaml
# model.yaml
estimator_class: sklearn.linear_model.Ridge
params:
  alpha: 0.5
```

This pattern is useful when:

- Different teams manage different pipeline components
- The same preprocessing config is shared across multiple pipelines
- Configs are generated programmatically and assembled at deployment time

## Allow Third-Party Modules

By default, only `sklearn` and `sklearn_wrap` classes can be resolved. Pass `trusted_modules` to allow additional packages:

```python
config.build(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "xgboost"}))
```

You can also set trusted modules globally:

```python
from sklearn_wrap.config import set_config

set_config(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "xgboost"}))
```

Or use a context manager for temporary scope:

```python
from sklearn_wrap.config import config_context

with config_context(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "xgboost"})):
    estimator = config.build()
```

The context manager is preferred for production code because it reverts the change automatically, preventing accidental trust escalation.

## See Also

- [Use YAML Configuration](yaml-configuration.md): core YAML workflow
- [Configuration Reference](../reference/configuration.md): `EstimatorConfig` API and config functions
- [About YAML Configuration Design](../explanation/yaml-config-design.md): why declarative config and the trusted modules security model
