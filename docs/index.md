![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Sklearn-Wrap's documentation

A Package for wrapping Python classes into Scikit-Learn estimators.

Sklearn-wrap enables you to wrap any Python class into a Scikit-Learn compatible estimator without rewriting your code. Whether you're integrating XGBoost's Booster API, custom gradient descent algorithms, or third-party machine learning libraries, Sklearn-Wrap provides the glue layer that makes them work seamlessly with Scikit-Learn's ecosystem.

With Sklearn-Wrap, you gain immediate access to GridSearchCV for hyperparameter tuning, meta estimators like Pipeline for composable workflows, and joblib for serialization, all while maintaining your original implementation. This enables data scientists to achieve Scikit-Learn compatibility without sacrificing custom logic or performance.

<div class="grid cards" markdown>

- **Get Started in 5 Minutes**

    ---

    Install Sklearn-Wrap and create your first wrapper. Learn the basic pattern and immediately gain meta estimator compatibility.

    [Getting Started](pages/getting-started.md)

- **Need Help?**

    ---

    Find answers and join the community. Ask questions, share experiences, and get help from maintainers and other users on GitHub Discussions.

    [GitHub Discussions](https://github.com/stateful-y/sklearn-wrap/discussions)

- **Learn the Concepts**

    ---

    Understand the wrapper pattern, parameter interface, and decorator mechanics. Dive deep into how Sklearn-Wrap integrates with Scikit-Learn's ecosystem.

    [User Guide](pages/user-guide.md)

- **See It In Action**

    ---

    Explore interactive examples from basics to advanced patterns. Each notebook demonstrates one concept with hands-on code you can modify and run interactively.

    [Examples](pages/examples.md)

</div>

## Table of Contents

### [Getting started](pages/getting-started.md)

Step-by-step guide to installing and setting up Sklearn-Wrap in your project.

- [Installation](pages/getting-started.md#installation)
- [Basic Usage](pages/getting-started.md#basic-usage)
- [Try Interactive Examples](pages/getting-started.md#try-interactive-examples)


### [Examples](pages/examples.md)

Interactive notebooks demonstrating Sklearn-Wrap features.

- [Quick Start](pages/examples.md#quick-start)
- [All Examples](pages/examples.md#examples)


### [User guide](pages/user-guide.md)

In-depth documentation on the design, architecture, and core concepts.

- [Core Concepts](pages/user-guide.md#core-concepts)
- [Configuration](pages/user-guide.md#configuration)
- [Best Practices](pages/user-guide.md#best-practices)

### [Reference](pages/api-reference.md)

Detailed reference for the Sklearn-Wrap API, including classes, functions, and configuration options.

## License

Sklearn-Wrap is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0). You are free to use, modify, and distribute this software under the terms of this license.
