# About YAML Configuration Design

Sklearn-Wrap's `EstimatorConfig` system lets you define estimator pipelines as YAML files instead of Python code. This page explains why declarative configuration exists, the security model behind trusted modules, and how config composition works.

## Why Declarative Configuration

Python code is the most flexible way to configure estimators, but it has drawbacks when configurations need to be shared, versioned, or managed by non-developers:

- **Python configs are executable**: A YAML file is inert data. It cannot import arbitrary modules, execute system commands, or produce side effects. This makes YAML configs safer to accept from untrusted sources (with the trusted modules guard).

- **Configs separate structure from logic**: A YAML file declares *what* the estimator looks like. The Python code declares *how* it behaves. This separation means data scientists can tune parameters in YAML files while engineers maintain the wrapper implementations in Python.

- **Configs are language-neutral**: YAML files can be generated or consumed by non-Python tools (CI pipelines, configuration management systems, web dashboards) without importing the Python package.

Declarative configuration is most valuable when the same estimator architecture needs multiple parameter variants (e.g., per-customer models, A/B test configurations, experiment tracking). When there is only one configuration that lives alongside the training code, Python is simpler and more direct.

## The Trusted Modules Security Model

`EstimatorConfig.build()` resolves dotted class paths (e.g., `sklearn.linear_model.Ridge`) into actual Python classes by importing them. This creates a code execution surface: a malicious YAML file could specify `os.system` or `subprocess.Popen` as the `estimator_class`.

To mitigate this, `EstimatorConfig` enforces a **trusted modules allowlist**. By default, only `sklearn` and `sklearn_wrap` are trusted. Any class path whose top-level module is not in the allowlist raises `UntrustedModuleError`.

The trust boundary is intentionally coarse-grained (top-level module, not individual classes) because:

- Fine-grained class-level allowlists are brittle and hard to maintain as libraries evolve
- The top-level module is the unit of trust in Python's packaging ecosystem. If you trust `xgboost`, you trust all public classes within it.
- Users explicitly opt in per-module, making the security posture visible and auditable

Three mechanisms control the allowlist:

1. **Per-call**: `config.build(trusted_modules=frozenset({...}))` for explicit, single-build scope
2. **Global**: `set_config(trusted_modules=...)` for persistent process-lifetime changes
3. **Context manager**: `config_context(trusted_modules=...)` for temporary scope that reverts automatically

The global and context manager approaches follow Scikit-Learn's own `set_config` / `config_context` pattern, keeping the API familiar.

## Config Composition with `!include`

Complex pipelines often share sub-configurations. The `!include` YAML tag lets you factor out reusable pieces into separate files:

```yaml
# pipeline.yaml
estimator_class: sklearn.pipeline.Pipeline
params:
  steps:
    - - scaler
      - !include preprocessing.yaml
    - - model
      - !include model.yaml
```

Paths in `!include` resolve relative to the including file, which means:

- Configs are portable, so moving a directory preserves all relative includes
- Shared components (e.g., a standard `preprocessing.yaml`) can live in a common directory
- Circular includes are detected and raise an error

This is similar to how Ansible uses `!include` for role composition, or how Docker Compose uses `extends`. The pattern trades some visibility (you must open multiple files to see the full config) for reusability and single-source-of-truth for shared components.

## YAML Anchors vs. `!include`

YAML natively supports anchors (`&name`) and merge keys (`<<: *name`) for reusing values within a single file. `!include` operates across files. They serve different needs:

**Anchors** work best for shared parameter blocks within one config file, such as a set of default hyperparameters reused across multiple estimators in the same pipeline definition.

**`!include`** works best for factoring out entire estimator configs that are shared across multiple pipeline files or managed by different teams.

Both can be combined: an included file can use anchors internally, and a parent file can anchor an included result.

## Relationship to Python Configuration

`EstimatorConfig` complements, rather than replaces, Python-based configuration. The typical workflow moves between both:

1. **Develop** in Python: Define wrappers, test configurations interactively
2. **Capture** to YAML: `EstimatorConfig.from_estimator(pipe).to_yaml("config.yaml")`
3. **Deploy** from YAML: `EstimatorConfig.from_yaml("config.yaml").build().fit(X, y)`
4. **Iterate**: Edit YAML parameters, re-deploy without code changes

The `from_estimator()` method ensures that any estimator you can build in Python can be serialized to YAML, and `from_yaml()` ensures any valid YAML config produces a working estimator. This round-trip guarantee is what makes the two approaches interchangeable.

## See Also

- [How to Use YAML Configuration](../how-to/yaml-configuration.md): core YAML workflow
- [How to Use Advanced YAML Patterns](../how-to/yaml-advanced.md): anchors, `!include`, multi-file configs
- [Configuration Reference](../reference/configuration.md): `EstimatorConfig` API and config functions
- [The Delegation Pattern](delegation-pattern.md): how wrappers and configs fit together
