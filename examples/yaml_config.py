# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "sklearn-wrap[config]",
# ]
# ///
"""
# YAML Configuration

Define, save, and load scikit-learn estimator configurations as YAML files.
Use YAML anchors for shared defaults and `!include` for multi-file composition.
"""

import marimo

__generated_with = "0.19.8"
__gallery__ = {
    "title": "YAML Configuration",
    "description": "Save and load estimator configurations as validated YAML files with inheritance support.",
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

    from sklearn_wrap.config import EstimatorConfig

    return EstimatorConfig, Path, Pipeline, Ridge, StandardScaler, np, tempfile, textwrap


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What You'll Learn

    - How to define estimator configurations as YAML
    - How to build estimators from YAML configs
    - How to capture an existing estimator's configuration
    - How YAML anchors and merge keys enable shared defaults
    - How `!include` composes configs from multiple files
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Building an Estimator from a Config

    `EstimatorConfig` holds a dotted import path and constructor parameters.
    Call `.build()` to get a ready-to-use estimator.
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
    ## 2. Capturing a Config from an Existing Estimator

    `EstimatorConfig.from_estimator()` inspects any sklearn-compatible estimator
    and produces a config that can reproduce it.
    """)
    return


@app.cell
def _(EstimatorConfig, Ridge):
    original = Ridge(alpha=5.0, fit_intercept=False)

    captured = EstimatorConfig.from_estimator(original)
    print(f"estimator_class: {captured.estimator_class}")
    print(f"params: {captured.params}")
    return captured, original


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. YAML Roundtrip

    Save a config to YAML with `.to_yaml()`, load it back with `.from_yaml()`,
    then `.build()` to get an identical estimator.
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
    ## 4. YAML Anchors and Merge Keys

    Native YAML anchors (`&name`) and merge keys (`<<: *name`) let you define
    shared parameter defaults and override them per estimator - no custom syntax needed.
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
    ## 5. Multi-file Composition with `!include`

    Split complex configurations across files. The `!include` tag loads and
    inlines another YAML file, with paths resolved relative to the including file.
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
    ## 6. Fit and Predict from YAML Config

    A complete workflow: load config, build estimator, fit, predict.
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
    ## Security

    By default, only classes from `sklearn` and `sklearn_wrap` can be resolved.
    Pass `trusted_modules` to `build()` to allow additional packages:

    ```python
    config.build(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "xgboost"}))
    ```

    This prevents arbitrary code execution from untrusted YAML files.
    """)
    return


if __name__ == "__main__":
    app.run()
