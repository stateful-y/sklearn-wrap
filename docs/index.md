![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Sklearn-Wrap's documentation

Sklearn-wrap enables you to wrap any Python class into a Scikit-Learn compatible estimator without rewriting your code. Whether you're integrating XGBoost's Booster API, custom gradient descent algorithms, or third-party machine learning libraries, Sklearn-Wrap provides the glue layer that makes them work seamlessly with Scikit-Learn's ecosystem.

With Sklearn-Wrap, you gain immediate access to GridSearchCV for hyperparameter tuning, meta estimators like Pipeline for composable workflows, joblib for serialization, and declarative YAML configuration via `EstimatorConfig` while maintaining your original implementation. This enables data scientists to achieve Scikit-Learn compatibility without sacrificing custom logic or performance.

<div class="grid cards" markdown>

- **Get Started in 5 Minutes**

    ---

    Install Sklearn-Wrap and create your first wrapper. Learn the basic pattern and immediately gain meta estimator compatibility.

    [Getting Started](pages/tutorials/getting-started.md)

- **How-to Guides**

    ---

    Task-oriented guides for wrapping classes, validating parameters, YAML configuration, GridSearchCV integration, and nesting wrappers.

    [Browse Guides](pages/how-to/index.md)

- **Understand the Design**

    ---

    Learn why the delegation pattern works, how `_fit_context` manages the lifecycle, and the trade-offs of composition over inheritance.

    [Core Concepts](pages/explanation/index.md)

- **API Reference**

    ---

    Complete reference for all Sklearn-Wrap classes, functions, and configuration options.

    [API Reference](pages/reference/index.md)

</div>

## License

Sklearn-Wrap is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0). You are free to use, modify, and distribute this software under the terms of this license.
