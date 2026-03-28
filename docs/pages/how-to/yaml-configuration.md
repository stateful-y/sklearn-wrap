# How to Use YAML Configuration

This guide shows you how to save, load, and validate estimator configurations as YAML files using `EstimatorConfig`. Use this when you want to manage estimator parameters as declarative config files rather than Python code.

## Prerequisites

Install the config extra:

```bash
pip install sklearn-wrap[config]
```

## Build an Estimator from a Config

Create an `EstimatorConfig` with a dotted import path and constructor parameters, then call `.build()`:

```python
from sklearn_wrap.config import EstimatorConfig

config = EstimatorConfig(
    estimator_class="sklearn.linear_model.Ridge",
    params={"alpha": 2.0, "fit_intercept": True},
)

estimator = config.build()
```

## Capture a Config from an Existing Estimator

Use `EstimatorConfig.from_estimator()` to snapshot any sklearn-compatible estimator:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
config = EstimatorConfig.from_estimator(pipe)
```

## Save and Load YAML

```python
# Save
config.to_yaml("pipeline.yaml")

# Load
loaded = EstimatorConfig.from_yaml("pipeline.yaml")
estimator = loaded.build()
```

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

## Allow Third-party Modules

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

## See Also

- [YAML Configuration example](/examples/yaml_config/) - interactive walkthrough
- [Configuration Reference](../reference/configuration.md) - `EstimatorConfig` API and config functions
- [About Core Concepts](../explanation/concepts.md) - how config fits into the wrapper pattern
