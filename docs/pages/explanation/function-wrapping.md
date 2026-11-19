# About Function Wrapping

Sklearn-Wrap provides two complementary approaches for integrating custom code with scikit-learn's ecosystem: `BaseClassWrapper` for wrapping classes, and `FunctionWrapper` for wrapping standalone callables. This page explains the functor design behind `FunctionWrapper`, the conventions it follows, and how it relates to its class-wrapping sibling.

## The Functor Approach

A functor is an object that can be called like a function but also carries state. `FunctionWrapper` turns any Python callable into a functor whose configuration is managed through scikit-learn's `get_params`/`set_params` protocol.

This is useful when you have a function (not a class) whose behavior you want to tune, compose in pipelines, or search over with `GridSearchCV`. Rather than manually threading configuration through function arguments, the wrapper stores config params and injects them at call time.

The core interaction is simple: positional arguments at the call site are data, stored keyword-only parameters are config.

```python
# Without a wrapper: config mixed with data at every call site
result = predict(X, scale=2.0, offset=1.0)

# With a wrapper: config stored once, data passed at call time
predictor = ScaledPredictor(fn=predict, scale=2.0, offset=1.0)
result = predictor(X)
```

## The Keyword-Only Convention

`FunctionWrapper` needs a way to distinguish data parameters (passed at call time) from config parameters (stored on the wrapper). It uses Python's keyword-only parameter syntax as the dividing line.

In a function signature, everything before the `*` separator is positional (data). Everything after is keyword-only (config):

```python
def my_function(X, y, *, alpha=1.0, beta=2.0):
    #             ↑ data    ↑ config
    ...
```

This convention was chosen because it aligns with how Python already distinguishes parameter kinds, requires no metadata or decorators, and makes the separation visible in the function signature itself. Functions that follow this convention work with `FunctionWrapper` without modification.

When `FunctionWrapper` inspects a callable's signature, it extracts only `KEYWORD_ONLY` parameters. `POSITIONAL_ONLY`, `POSITIONAL_OR_KEYWORD`, and `VAR_POSITIONAL` parameters are ignored entirely, since they represent data that flows through `__call__`.

If the function accepts `**kwargs`, the wrapper allows arbitrary extra config parameters beyond the declared keyword-only ones. This is useful for functions that forward configuration to underlying systems.

## Relationship to BaseClassWrapper

`FunctionWrapper` and `BaseClassWrapper` are independent siblings, both inheriting from scikit-learn's `BaseEstimator`. They share the same parameter management philosophy but differ in their lifecycle:

| Aspect | BaseClassWrapper | FunctionWrapper |
|--------|-----------------|-----------------|
| Wraps | A class | A callable |
| Lifecycle | Creates `self.instance_` on fit | No instance lifecycle |
| Config stored as | `self.params` | `self._params` |
| Delegation | `self.instance_.method(...)` | `self.callable_fn(*args, **self._params)` |
| Call convention | Not callable | Callable via `__call__` |

`BaseClassWrapper` manages a three-phase lifecycle (configuration, instantiation, delegation) because it wraps classes that need to be constructed, fitted, and queried. `FunctionWrapper` is simpler: the callable exists at construction time and is invoked directly. There is no instance to create or destroy.

Both wrappers can be extended into full sklearn estimators by adding mixins (`RegressorMixin`, `ClassifierMixin`) and implementing `fit`/`predict`/`transform`. The `_fit_context` decorator works identically with both.

## Internal Parameter Storage

`FunctionWrapper` stores config parameters in `self._params` (with an underscore prefix) rather than `self.params`. This is a deliberate choice to avoid a `RecursionError` in scikit-learn's `BaseEstimator.__repr__`.

The issue: sklearn's repr system introspects the `__init__` signature to detect changed parameters. Since `FunctionWrapper.__init__` accepts `**params`, sklearn cannot determine individual parameter defaults and falls back to inspecting instance attributes. A public `self.params` dict attribute triggers a repr cycle where sklearn tries to display the dict, encounters the estimator again, and recurses.

`BaseClassWrapper` avoids this because its `__init__` has explicit attribute access patterns that sklearn's repr can resolve. `FunctionWrapper` with `__init__(**params)` does not have those, so the underscore prefix hides the dict from sklearn's introspection.

## The `instantiate()` Step

Both wrappers provide an `instantiate()` method, but they do different things:

- **BaseClassWrapper**: validates parameters, creates `self.instance_ = estimator_class(**params)`, and resets the fitted flag.
- **FunctionWrapper**: validates parameters and checks for required sentinels. No instance is created, and the fitted flag is not touched.

`FunctionWrapper.instantiate()` is called automatically by `__call__` (before invoking the callable) and by `_fit_context` (before running `fit`). It acts as a validation gate, ensuring that all required parameters have been provided before the callable is invoked.

## Extending to a Full Estimator

`FunctionWrapper` on its own is a parameter-managed functor. To turn it into a proper scikit-learn estimator, subclass it with a mixin and implement the estimator interface:

```python
class FunctionRegressor(FunctionWrapper, RegressorMixin):
    _callable_name = "fn"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return self.callable_fn(np.asarray(X), **self._params)
```

The `_fit_context` decorator is reused without modification. It checks for the presence of `instantiate()` on the estimator and calls it before fit, regardless of whether the estimator wraps a class or a function.

## Serialization Considerations

Named functions defined at module level serialize correctly with `pickle` and `joblib`, since Python serializes them by reference (module path + function name). Lambdas and closures cannot be serialized this way, because they lack a stable qualified name. If you need to save and load a `FunctionWrapper`, use named functions.

`functools.partial` objects work correctly with `FunctionWrapper` (the signature is resolved through the partial), but they are deep-copied during `clone()` rather than preserved by reference. This means identity checks like `cloned.callable_fn is original_partial` will fail, but behavioral equivalence is maintained.
