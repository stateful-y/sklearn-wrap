# About Trade-offs and Limitations

The delegation approach that Sklearn-Wrap uses comes with trade-offs worth understanding. This page discusses the constraints, their rationale, and when they matter in practice.

## Estimator Class Immutability

Once a wrapper is instantiated, the wrapped class cannot be changed via `set_params()`. This is by design because changing the class would invalidate all parameter metadata, since the new class might accept entirely different constructor arguments.

Scikit-Learn's `get_params()` contract requires that the returned dictionary fully describes how to reconstruct the estimator. If the wrapped class could change, the parameter dictionary would become inconsistent with the actual object state, breaking `clone()`, `GridSearchCV`, and any tool that relies on parameter introspection.

To wrap a different class, create a new wrapper instance. This is consistent with how Scikit-Learn handles estimator identity: a `Ridge` estimator does not become a `Lasso` via `set_params()`.

## Parameter Validation Overhead

Sklearn-Wrap validates parameters before every instantiation, adding overhead per `fit()` call. This overhead is negligible for typical training workflows but becomes noticeable in two scenarios:

- **Tight training loops** where `fit()` is called thousands of times (e.g., online learning)
- **Large-scale hyperparameter searches** where `GridSearchCV` spawns hundreds of clones

For these cases, `sklearn.set_config(skip_parameter_validation=True)` disables validation across the entire Scikit-Learn ecosystem after you have verified parameters during development. The `prefer_skip_nested_validation=True` flag in `@_fit_context` also helps by skipping redundant validation in nested estimator hierarchies.

## No Automatic Metadata Routing

The wrapper can consume metadata like `sample_weight` in its `fit()` method, but cannot automatically route metadata to nested wrapped estimators. You must handle routing manually if needed.

This is a conscious boundary. Automatic routing across heterogeneous interfaces would be fragile because the wrapper cannot know which metadata keys the wrapped class expects, especially when method names are remapped (e.g., `fit_model` instead of `fit`). Scikit-Learn's own metadata routing (introduced in 1.3) operates within its known interface contract; wrappers sit outside that contract by design.

For pipelines where metadata routing matters, consider exposing the metadata as explicit parameters or implementing `__sklearn_tags__` on the wrapper class.

## Flat vs. Deep Parameter Hierarchies

When nesting wrappers, you choose between flat and deep parameter structures. Both work, but they have different ergonomics:

**Flat structures** put all knobs at one level. They are simpler to configure and search over with `GridSearchCV`, but obscure the model architecture. If two nested components share a parameter name (e.g., both have `alpha`), you must disambiguate.

**Deep hierarchies** reflect the model architecture explicitly. The double-underscore syntax (`outer__inner__alpha`) keeps parameters namespaced, and you can swap sub-components independently. The cost is longer parameter names and more complex grid definitions.

Neither approach is universally better. Flat structures suit simple compositions; deep hierarchies suit modular architectures where components are developed and tested independently.

## The Wrapper is Not the Wrapped Class

Wrappers proxy attribute access to `self.instance_` after fitting, but they are not transparent proxies. Some differences to be aware of:

- `isinstance(wrapper, WrappedClass)` returns `False` because the wrapper is a separate type
- Attributes set on the wrapper during `__init__` are wrapper attributes, not instance attributes
- If the wrapped class uses class-level attributes or metaclasses, those do not propagate through the wrapper

For most Scikit-Learn workflows (fit/predict/transform, parameter inspection, serialization), this distinction does not matter. It becomes relevant when third-party code inspects the estimator's actual type rather than its interface.

## See Also

- [The Delegation Pattern](delegation-pattern.md): architecture and three-phase lifecycle
- [The Fit Context Lifecycle](fit-context-lifecycle.md): validation flow and `prefer_skip_nested_validation`
- [How to Nest Wrappers](../how-to/nest-wrappers.md): practical guide to nested hierarchies
- [Configuration Reference](../reference/configuration.md): all wrapper class attributes
