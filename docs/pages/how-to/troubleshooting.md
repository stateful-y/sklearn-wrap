# Troubleshooting

Common errors when working with Sklearn-Wrap, their causes, and how to fix them.

## Wrapper Definition Errors

### Missing `_estimator_name`

```text
ValueError: Class should define a static '_estimator_name'.
```

**Cause**: The wrapper class does not define `_estimator_name` as a string class attribute.

**Fix**: Add a string attribute to your wrapper class:

```python
class MyWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "regressor"
    _estimator_base_class = object
```

### Missing `_estimator_base_class`

```text
ValueError: Class should define a static '_estimator_base_class'.
```

**Cause**: The wrapper class does not define `_estimator_base_class`.

**Fix**: Add a class attribute. Use `object` to accept any class, or a specific base class for stricter validation:

```python
_estimator_base_class = object  # accepts any class
```

## Constructor Errors

### Missing Required Parameter

```text
TypeError: MyWrapper.__init__() missing required keyword argument: 'regressor'
```

**Cause**: The wrapped class was not passed when creating the wrapper, and no `_estimator_default_class` is defined.

**Fix**: Pass the class to wrap using the keyword matching `_estimator_name`:

```python
wrapper = MyWrapper(regressor=MyEstimator, alpha=1.0)
```

If the wrapper should have a sensible default, define `_estimator_default_class`:

```python
class MyWrapper(BaseClassWrapper, RegressorMixin):
    _estimator_name = "regressor"
    _estimator_base_class = object
    _estimator_default_class = MyDefaultEstimator
```

### Estimator Is Not a Class

```text
TypeError: regressor parameter for estimator MyWrapper is not a class. It is <MyEstimator object>.
```

**Cause**: An instance was passed instead of a class. The wrapper expects a class (not an instantiated object) because it handles instantiation itself during `fit()`.

**Fix**: Pass the class, not an instance:

```python
# Wrong
wrapper = MyWrapper(regressor=MyEstimator())

# Correct
wrapper = MyWrapper(regressor=MyEstimator)
```

### Wrong Base Class

```text
ValueError: Invalid regressor class WrongClass for estimator MyWrapper.
Valid estimator class should be derived from ExpectedBase.
```

**Cause**: The passed class does not inherit from `_estimator_base_class`.

**Fix**: Either pass a class that inherits from the expected base, or widen the constraint:

```python
_estimator_base_class = object  # accepts any class
```

## Parameter Errors

### Reserved Delimiter in Parameter Name

```text
ValueError: Parameter name my__param cannot contain '__' (double underscore).
This delimiter is reserved for nested parameter syntax.
```

**Cause**: The wrapped class has a constructor parameter containing `__`, which conflicts with Scikit-Learn's nested parameter syntax.

**Fix**: Rename the parameter in the wrapped class, or create an adapter class that translates the parameter name:

```python
class Adapter:
    def __init__(self, my_param=1.0):
        self._original = OriginalClass(my__param=my_param)
```

### Invalid Parameter Name

```text
ValueError: my_param is not a valid parameter for class MyEstimator.
```

**Cause**: A keyword argument was passed to the wrapper that does not match any constructor parameter of the wrapped class.

**Fix**: Check the wrapped class's `__init__` signature for valid parameter names.

### Cannot Change Estimator Class

```text
ValueError: Cannot change estimator class via set_params.
The 'regressor' parameter cannot be set. Redeclare the estimator...
```

**Cause**: `set_params()` was called trying to change the wrapped class itself.

**Fix**: Create a new wrapper instance with the different class:

```python
# Wrong
wrapper.set_params(regressor=DifferentClass)

# Correct
new_wrapper = MyWrapper(regressor=DifferentClass, alpha=1.0)
```

## Fit-Time Errors

### Required Parameter Still Unset

```text
ValueError: Class MyEstimator requires parameter alpha.
```

**Cause**: The wrapped class has a required constructor parameter (no default value) that was not provided when creating the wrapper.

**Fix**: Pass all required parameters:

```python
wrapper = MyWrapper(regressor=MyEstimator, alpha=1.0)
```

### Unfitted Estimator

```text
sklearn.exceptions.NotFittedError: This MyWrapper instance is not fitted yet.
```

**Cause**: `predict()` or `transform()` was called before `fit()`.

**Fix**: Call `fit()` first:

```python
wrapper.fit(X_train, y_train)
predictions = wrapper.predict(X_test)
```

## Nested Parameter Errors

### Invalid Nested Parameter

```text
ValueError: Invalid parameter inner_alpha for estimator MyWrapper.
Valid parameters are: ['regressor', 'alpha', 'beta']
```

**Cause**: The base key of a nested parameter (before the first `__`) does not match any wrapper parameter.

**Fix**: Check `wrapper.get_params()` for valid parameter names. Nested parameters use `__` to separate levels:

```python
# Set a nested parameter
wrapper.set_params(inner__alpha=2.0)
```

### Nested Object Missing `set_params`

```text
AttributeError: Cannot set nested parameters on inner.
Object of type int does not have a set_params method.
```

**Cause**: Double-underscore syntax was used to reach into a parameter that is not itself an estimator or wrapper.

**Fix**: Only use `__` syntax for parameters that are `BaseClassWrapper` instances or Scikit-Learn estimators with `set_params()`.

## YAML Configuration Errors

### Untrusted Module

```text
UntrustedModuleError: Module 'xgboost' is not in the trusted modules list.
```

**Cause**: The YAML config references a class from a module not in the trusted allowlist.

**Fix**: Add the module to trusted modules:

```python
from sklearn_wrap.config import set_config

set_config(trusted_modules=frozenset({"sklearn", "sklearn_wrap", "xgboost"}))
```

## See Also

- [How to Validate Parameters](validate-parameters.md): adding parameter constraints
- [How to Test Your Wrapper](test-wrappers.md): verifying wrapper correctness
- [Configuration Reference](../reference/configuration.md): all wrapper class attributes
