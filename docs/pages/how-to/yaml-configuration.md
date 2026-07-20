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
