# About the Fit Context Lifecycle

The `@_fit_context` decorator is the mechanism that bridges wrapper configuration and the wrapped instance. This page explains what it does, why it exists, and the validation trade-offs it manages.

## What the Decorator Does

The `@_fit_context` decorator wraps your `fit()` method to handle three responsibilities automatically:

1. **Instantiation**: Calls `instantiate()`, which validates parameters and creates `self.instance_` by calling `estimator_class(**params)`. This is the transition from the configuration phase to a live object.

2. **Parameter validation**: Runs Scikit-Learn's parameter validation if `_parameter_constraints` are defined on the wrapper class. This catches invalid parameter values before they reach the wrapped class.

3. **Validation context management**: Sets up a context that nested estimators can check to decide whether they should validate their own parameters.

Without the decorator, you would need to call `self.instantiate()` manually as the first line of every `fit()` method. The decorator eliminates this boilerplate and ensures consistent behavior across all wrappers.

## Why Deferred Instantiation

The wrapped instance is not created in `__init__()` but deferred until `fit()`. This is a deliberate design choice driven by how Scikit-Learn works internally.

Scikit-Learn's `clone()` function creates new estimator instances by reading `get_params()` and passing them to the constructor. `GridSearchCV` relies on this to create fresh copies for each parameter combination it evaluates. If the wrapped instance were created eagerly in `__init__()`, cloning would create the instance twice: once during construction, then thrown away and recreated at `fit()` time.

Deferring instantiation to `fit()` means:

- `clone()` produces lightweight wrapper objects with just parameter dictionaries
- `GridSearchCV` can efficiently spawn hundreds of clones without constructing heavy objects
- Parameter changes via `set_params()` take effect on the next `fit()` call naturally

## The `prefer_skip_nested_validation` Parameter

The `prefer_skip_nested_validation` parameter reflects a practical trade-off between safety and performance.

**For most wrappers (`prefer_skip_nested_validation=True`)**: Parameters come directly from the user or from `GridSearchCV`. Validating them once at the outer level is sufficient. Nested wrappers can skip redundant checks, which matters when `GridSearchCV` is fitting hundreds of estimator copies.

**For meta-estimators (`prefer_skip_nested_validation=False`)**: The wrapper constructs or transforms parameters internally before passing them to nested estimators. In this case, the nested estimators should validate their inputs because the meta-estimator may produce invalid combinations.

The naming follows Scikit-Learn's convention. Setting `prefer_skip_nested_validation=True` does not unconditionally disable validation. It sets a preference that nested estimators honor when running inside a parent `fit()` context. Outside of any context (e.g., calling `fit()` directly), validation always runs.

## Using with `partial_fit`

The decorator can also be applied to `partial_fit()` for incremental learning. On the first call, `instantiate()` creates the wrapped instance; on subsequent calls, it reuses the existing `self.instance_`. This allows the wrapped class to maintain state across multiple `partial_fit()` invocations.

The key difference from `fit()` is that `partial_fit()` should not recreate the instance on every call. The decorator handles this by checking whether `self.instance_` already exists.

## Disabling Validation Globally

For performance-critical applications where parameter validation overhead matters (tight training loops, large-scale hyperparameter searches), Scikit-Learn provides a global switch:

```python
import sklearn
sklearn.set_config(skip_parameter_validation=True)
```

This disables all parameter validation across the entire Scikit-Learn ecosystem, including Sklearn-Wrap's constraint checking. Use this only after verifying parameters are correct during development.

## See Also

- [How to Wrap a Class](../how-to/wrap-a-class.md): using `@_fit_context` in practice
- [How to Validate Parameters](../how-to/validate-parameters.md): defining parameter constraints
- [The Delegation Pattern](delegation-pattern.md): the three-phase lifecycle that `_fit_context` automates
- [API Reference](../reference/api.md): `_fit_context` signature and options
