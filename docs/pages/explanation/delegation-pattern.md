# About the Delegation Pattern

Sklearn-Wrap bridges arbitrary Python classes with Scikit-Learn's estimator interface through a thin delegation layer. This page explains the core architectural approach, the reasoning behind it, and the configuration that makes it work.

## The Problem

Machine learning practitioners frequently have working implementations (custom algorithms, third-party libraries like XGBoost's low-level Booster API, or legacy research code) but want to use Scikit-Learn's powerful tooling: `GridSearchCV` for hyperparameter tuning, `Pipeline` for composable workflows, `cross_val_score` for validation, and `joblib` for serialization.

The traditional solution is rewriting the implementation to inherit from Scikit-Learn's `BaseEstimator`. This requires understanding Scikit-Learn's internal conventions (`get_params`/`set_params`, constructor parameter rules, fitted attribute naming) and modifying the original code, which can be time-consuming and error-prone, especially for third-party code you don't control.

## Composition Over Inheritance

Sklearn-Wrap takes a different approach: composition over inheritance. Rather than forcing your class to inherit from Scikit-Learn base classes, it creates a **wrapper layer** that delegates to your original implementation unchanged.

This is analogous to the Adapter pattern in software design. Your wrapper class acts as a translator between two interfaces: Scikit-Learn's estimator protocol on one side, and your class's custom API on the other. The original class never needs to know about Scikit-Learn at all.

The approach is particularly effective when:

- You are integrating a third-party library whose source you cannot modify
- You have battle-tested code whose internals you prefer not to touch
- The wrapped class uses non-standard method names (like `fit_model` instead of `fit`)
- You want the same class to be usable both inside and outside Scikit-Learn workflows
- Multiple classes share a common base, so a single wrapper covers the entire family

For new implementations where you control the entire codebase, inheriting directly from Scikit-Learn's `BaseEstimator` is simpler and more direct. Sklearn-Wrap adds value specifically when that direct approach is impractical.

## The Three-Phase Lifecycle

At its heart, Sklearn-Wrap uses a separation between the wrapper's configuration and the wrapped instance:

1. **Configuration phase**: The wrapper stores the class to wrap and its constructor parameters as a flat dictionary (`params`). Scikit-Learn's `get_params()` and `set_params()` operate on this dictionary, enabling tools like `GridSearchCV` to manipulate parameters without touching the wrapped class.

2. **Instantiation phase**: When `fit()` is called, the wrapper creates the actual instance by calling `estimator_class(**params)` and stores it as `self.instance_`. This deferred instantiation is critical because it allows Scikit-Learn to clone estimators and modify parameters between fits.

3. **Delegation phase**: The wrapper's `fit()`, `predict()`, and other methods delegate to `self.instance_`, translating between Scikit-Learn's expected interface and the wrapped class's actual methods.

This three-phase lifecycle is what makes the wrapper transparent to Scikit-Learn. From the outside, it behaves exactly like a native estimator.

## Required Configuration

Every wrapper class must define two attributes that configure the delegation:

**`_estimator_name`** controls the keyword argument name used to pass the wrapped class. If you set `_estimator_name = "regressor"`, users create wrappers with `MyWrapper(regressor=SomeClass, ...)`. This name also appears as a key in `get_params()` output, keeping the wrapped class discoverable and configurable.

**`_estimator_base_class`** validates that the wrapped class inherits from an expected base. This catches configuration errors early: passing an incompatible class raises an error at wrapper creation time rather than failing deep inside `fit()`. Use `object` for minimal validation, or a specific base class for stricter type checking.

Setting `_estimator_base_class` to a shared base class also enables a single wrapper to serve an entire family of implementations. If several classes share a common base (e.g., multiple XGBoost callback types all inherit from `xgb.callback.TrainingCallback`), one wrapper targeting that base class can wrap any of them. Users swap implementations by passing a different class at construction time, without writing additional wrapper code.

## Nested Parameters and the `__` Syntax

Scikit-Learn's double-underscore syntax (`outer__inner__param`) enables reaching into nested estimator hierarchies. Sklearn-Wrap supports this natively through `get_params(deep=True)` and `set_params()`.

When a wrapper parameter is itself a `BaseClassWrapper` instance, `get_params(deep=True)` recurses into it, prefixing each nested parameter with the outer parameter name. This creates a flat dictionary that `GridSearchCV` can iterate over for hyperparameter search across the entire hierarchy.

Some teams prefer flat parameter spaces with all knobs at one level, while others build deep hierarchies reflecting the model architecture. Both approaches work with Sklearn-Wrap. Flat structures are simpler to configure; nested structures make the architecture explicit and allow swapping sub-components independently.

## Connections

- The wrapper pattern is an instance of the [Adapter pattern](https://en.wikipedia.org/wiki/Adapter_pattern), bridging incompatible interfaces
- Scikit-Learn's own `Pipeline` and `VotingClassifier` use similar delegation internally
- The `EstimatorConfig` system extends this further by enabling YAML-based configuration and composition of wrapped estimators

## See Also

- [How to Wrap a Class](../how-to/wrap-a-class.md): step-by-step guide for regressor, classifier, and transformer wrappers
- [How to Nest Wrappers](../how-to/nest-wrappers.md): composing wrappers hierarchically
- [The Fit Context Lifecycle](fit-context-lifecycle.md): how `_fit_context` automates instantiation
- [API Reference](../reference/api.md): full `BaseClassWrapper` documentation
