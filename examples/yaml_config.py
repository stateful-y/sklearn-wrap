# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "sklearn-wrap[config]",
# ]
# ///
"""
# How to Configure Estimators with YAML

Define, save, and load scikit-learn estimator configurations as validated YAML files.
"""

import marimo

__generated_with = "0.19.8"
__gallery__ = {
    "title": "How to Configure Estimators with YAML",
    "description": "Define, save, and load estimator configurations as validated YAML files.",
    "category": "how-to",
    "companion": "pages/how-to/yaml-configuration.md",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import tempfile
    import textwrap
    from pathlib import Path

    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_wrap.config import EstimatorConfig, config_context, get_config, set_config

    return EstimatorConfig, Path, Pipeline, Ridge, StandardScaler, config_context, get_config, np, set_config, tempfile, textwrap


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This guide covers the `EstimatorConfig` API: building estimators from YAML,
    capturing configs from existing estimators, composing multi-file configs with
    `!include`, and validating parameters at build time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Build an Estimator from a Config

    Create an `EstimatorConfig` with a dotted import path and parameters,
    then call `.build()` to get a ready-to-use estimator.
    """)
    return


@app.cell
def _(EstimatorConfig):
    config = EstimatorConfig(
        estimator_class="sklearn.linear_model.Ridge",
        params={"alpha": 2.0, "fit_intercept": True},
    )

    estimator = config.build()
    print(f"Built: {estimator}")
    print(f"alpha={estimator.alpha}, fit_intercept={estimator.fit_intercept}")
    return config, estimator


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Capture a Config from an Existing Estimator

    Use `EstimatorConfig.from_estimator()` to inspect any sklearn-compatible
    estimator and produce a config.

    The capture records only the parameters you set. Ones left at their
    constructor default are omitted, so the result reads like a config you
    would have written by hand. The cost is that an omitted parameter is no
    longer pinned: it takes whatever the installed scikit-learn makes the
    default. Pass `prune_defaults=False` when you need every value fixed.
    """)
    return


@app.cell
def _(EstimatorConfig, Ridge):
    original = Ridge(alpha=5.0, fit_intercept=False)

    captured = EstimatorConfig.from_estimator(original)
    print(f"estimator_class: {captured.estimator_class}")
    print(f"params: {captured.params}")

    pinned = EstimatorConfig.from_estimator(original, prune_defaults=False)
    print(f"params with prune_defaults=False: {len(pinned.params)} entries")
    return captured, original


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. YAML Roundtrip

    Save a config with `.to_yaml()`, load it with `.from_yaml()`, then
    `.build()` to reconstruct the estimator.
    """)
    return


@app.cell
def _(EstimatorConfig, Path, Ridge, tempfile):
    work_dir = Path(tempfile.mkdtemp())

    # Save
    est = Ridge(alpha=7.0)
    cfg = EstimatorConfig.from_estimator(est)
    yaml_path = work_dir / "ridge.yaml"
    cfg.to_yaml(yaml_path)
    print("Saved YAML:")
    print(yaml_path.read_text())

    # Load and rebuild
    loaded = EstimatorConfig.from_yaml(yaml_path)
    rebuilt = loaded.build()
    print(f"Rebuilt alpha: {rebuilt.alpha}")
    return cfg, est, loaded, rebuilt, work_dir, yaml_path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Use YAML Anchors for Shared Defaults

    Define shared parameter defaults with anchors (`&name`) and merge keys
    (`<<: *name`), then override per estimator.
    """)
    return


@app.cell
def _(EstimatorConfig, Path, tempfile):
    anchor_dir = Path(tempfile.mkdtemp())
    anchor_yaml = anchor_dir / "anchors.yaml"
    anchor_yaml.write_text(
        "_defaults: &defaults\n"
        "  fit_intercept: true\n"
        "  solver: auto\n"
        "\n"
        "estimator_class: sklearn.linear_model.Ridge\n"
        "params:\n"
        "  <<: *defaults\n"
        "  alpha: 0.5\n"
    )

    anchor_config = EstimatorConfig.from_yaml(anchor_yaml)
    anchor_est = anchor_config.build()
    print(f"alpha={anchor_est.alpha} (overridden)")
    print(f"fit_intercept={anchor_est.fit_intercept} (from defaults)")
    print(f"solver={anchor_est.solver!r} (from defaults)")
    return anchor_config, anchor_dir, anchor_est, anchor_yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Compose Configs with `!include`

    Split complex configurations across files. The `!include` tag loads another
    YAML file inline, with paths resolved relative to the including file.
    """)
    return


@app.cell
def _(EstimatorConfig, Path, tempfile, textwrap):
    include_dir = Path(tempfile.mkdtemp())

    # Write component configs
    (include_dir / "scaler.yaml").write_text(textwrap.dedent("""\
        estimator_class: sklearn.preprocessing.StandardScaler
        params:
          with_mean: true
          with_std: true
    """))

    (include_dir / "ridge.yaml").write_text(textwrap.dedent("""\
        estimator_class: sklearn.linear_model.Ridge
        params:
          alpha: 0.1
    """))

    # Write the pipeline config that includes them
    (include_dir / "pipeline.yaml").write_text(textwrap.dedent("""\
        estimator_class: sklearn.pipeline.Pipeline
        params:
          steps:
            - - scaler
              - !include scaler.yaml
            - - ridge
              - !include ridge.yaml
    """))

    pipeline_config = EstimatorConfig.from_yaml(include_dir / "pipeline.yaml")
    pipeline = pipeline_config.build()
    print(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
    print(f"Scaler with_mean: {pipeline.steps[0][1].with_mean}")
    print(f"Ridge alpha: {pipeline.steps[1][1].alpha}")
    return include_dir, pipeline, pipeline_config


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Fit and Predict from a YAML Config

    Load a config, build the estimator, fit, and predict - end to end.
    """)
    return


@app.cell
def _(EstimatorConfig, Path, np, tempfile, textwrap):
    fit_dir = Path(tempfile.mkdtemp())
    (fit_dir / "model.yaml").write_text(textwrap.dedent("""\
        estimator_class: sklearn.pipeline.Pipeline
        params:
          steps:
            - - scaler
              - estimator_class: sklearn.preprocessing.StandardScaler
                params: {}
            - - ridge
              - estimator_class: sklearn.linear_model.Ridge
                params:
                  alpha: 1.0
    """))

    model = EstimatorConfig.from_yaml(fit_dir / "model.yaml").build()

    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    X_test = np.array([[2, 3], [4, 5]])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    return X_test, X_train, fit_dir, model, predictions, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Validate Parameters

    `build()` validates parameter names against the target class constructor
    before instantiation. Typos in YAML are caught immediately.
    """)
    return


@app.cell
def _(EstimatorConfig, mo):
    bad_config = EstimatorConfig(
        estimator_class="sklearn.linear_model.Ridge",
        params={"alpha": 1.0, "allpha": 2.0},  # typo!
    )

    try:
        bad_config.build()
    except ValueError as exc:
        mo.output.replace(mo.md(f"**Caught at build time:** `{exc}`"))
    return (bad_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    If a class accepts `**kwargs` (like `Pipeline`), extra parameter names are
    allowed. You can also disable validation with `validate_params=False`.
    """)
    return


@app.cell
def _(EstimatorConfig):
    # Pipeline accepts **kwargs - no false positives
    pipe_cfg = EstimatorConfig(
        estimator_class="sklearn.pipeline.Pipeline",
        params={
            "steps": [
                ["ridge", {"estimator_class": "sklearn.linear_model.Ridge"}],
            ],
            "memory": None,
        },
    )
    pipe = pipe_cfg.build()
    print(f"Pipeline built with validation: {[name for name, _ in pipe.steps]}")
    return pipe, pipe_cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Manage Trusted Modules

    By default only `sklearn` and `sklearn_wrap` classes can be resolved.
    Use `set_config` to register packages globally, or `config_context` for
    a temporary scope.
    """)
    return


@app.cell
def _(EstimatorConfig, config_context, get_config, mo, set_config):
    # Check the defaults
    defaults = get_config()["trusted_modules"]
    mo.output.replace(mo.md(f"Default trusted modules: `{sorted(defaults)}`"))

    # Temporarily trust builtins
    with config_context(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "builtins"})):
        d = EstimatorConfig(estimator_class="builtins.dict").build()
        print(f"Inside context: built {type(d).__name__}")

    # Back to defaults outside the context
    print(f"After context: trusted modules = {sorted(get_config()['trusted_modules'])}")
    return d, defaults


if __name__ == "__main__":
    app.run()
