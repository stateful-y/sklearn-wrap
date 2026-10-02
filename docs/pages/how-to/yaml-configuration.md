# How to Use YAML Configuration

This guide shows you how to save, load, and validate estimator configurations as YAML files using `EstimatorConfig`. Use this when you want to manage estimator parameters as declarative config files rather than Python code.

## Prerequisites

Install the config extra:

```bash
pip install sklearn-wrap[config]
```

<!-- COMPANION_NOTEBOOKS -->

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

The captured config records only what you set. Parameters still at their
constructor default are omitted, at every level of nesting, so the result reads
like a config you would have written by hand:

```yaml
estimator_class: sklearn.pipeline.Pipeline
params:
  steps:
  - - scaler
    - estimator_class: sklearn.preprocessing.StandardScaler
      params: {}
  - - ridge
    - estimator_class: sklearn.linear_model.Ridge
      params:
        alpha: 0.5
```

That brevity has a cost. An omitted parameter is no longer pinned: the config
takes whatever the installed library makes the default. If a later scikit-learn
changes a default you were relying on, the rebuilt estimator changes with it,
silently. Where exact reproducibility matters more than readability, capture the
full set instead:

```python
config = EstimatorConfig.from_estimator(pipe, prune_defaults=False)
```

Class paths are recorded at their shortest public import path, so `Ridge` is
written as `sklearn.linear_model.Ridge` rather than the private module it is
defined in. Configs naming the private path still build, so files written before
this behavior existed keep working.

## Save and Load YAML

```python
# Save
config.to_yaml("pipeline.yaml")

# Load
loaded = EstimatorConfig.from_yaml("pipeline.yaml")
estimator = loaded.build()
```

## See Also

- [Advanced YAML Patterns](yaml-advanced.md): YAML anchors, `!include` composition, multi-file configs
- [YAML Configuration example](/examples/yaml_config/): interactive walkthrough
- [Configuration Reference](../reference/configuration.md): `EstimatorConfig` API and config functions
- [About YAML Configuration Design](../explanation/yaml-config-design.md): why declarative config and the trusted modules security model
