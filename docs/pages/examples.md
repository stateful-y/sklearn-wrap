# Examples

Learn Sklearn-Wrap through focused, interactive examples. Each notebook demonstrates one core concept in 5-15 minutes. Examples are organized from general concepts to specific use cases.

## Quick Start

New to Sklearn-Wrap? Start with the first three examples in sequence:

1. **first_wrapper.py** - Learn the basic wrapper pattern
2. **parameter_interface.py** - Understand parameter management
3. **grid_search.py** - Apply to hyperparameter tuning

## Examples

### [first_wrapper.py](/examples/first_wrapper/)

**Creating Your First Sklearn-Compatible Wrapper**

Start here to understand the fundamental wrapper pattern. This example walks through wrapping a custom polynomial regression algorithm that uses gradient descent. You'll learn the three essential components every wrapper needs: the `_estimator_name` and `_estimator_base_class` attributes, plus the `@_fit_context` decorator. By the end, you'll have a working wrapper that integrates seamlessly with Scikit-Learn's ecosystem.

### [parameter_interface.py](/examples/parameter_interface/)

**Mastering the Parameter Interface** (8 minutes)

Dive deep into how `get_params()` and `set_params()` work under the hood. This example demonstrates why Scikit-Learn needs this interface for GridSearchCV and Pipeline, and shows exactly what happens when you change parameters on a fitted estimator. You'll see interactive demos of parameter inspection and modification, plus understand the distinction between wrapper parameters and wrapped class parameters. Essential for working with Scikit-Learn's hyperparameter tuning tools.

### [validation.py](/examples/validation/)

**Error Patterns and Parameter Validation**

Learn to handle errors gracefully by exploring five common validation scenarios: invalid parameters, wrong base class inheritance, missing required parameters, reserved double-underscore syntax, and parameter constraint violations. Each pattern includes a live error demonstration showing exactly what breaks and why. Understanding these patterns helps you debug wrapper issues quickly and write more robust wrappers with proper `_parameter_constraints`.

### [grid_search.py](/examples/grid_search/)

**Hyperparameter Tuning with GridSearchCV**

Apply what you've learned to real hyperparameter optimization. This example wraps a k-nearest neighbors classifier and uses GridSearchCV to find optimal `n_neighbors` and `weights` values through cross-validation. You'll see how Scikit-Learn's parameter interface enables automatic tuning without any special code in your wrapper. Includes interactive parameter exploration and visual comparison of grid search results versus default parameters.

### [xgboost_wrapper.py](/examples/xgboost_wrapper/)

**Wrapping XGBoost's Booster API**

Integrate third-party libraries by wrapping XGBoost's low-level training API. Unlike XGBoost's built-in Scikit-Learn wrappers (XGBRegressor/XGBClassifier), the Booster API offers finer control over the training process. This example shows how to handle library-specific quirks like DMatrix conversion and parameter formatting. Demonstrates that Sklearn-Wrap isn't just for custom code, it's equally valuable for bringing external libraries into Scikit-Learn's ecosystem.

### [serialization.py](/examples/serialization/)

**Persistence with Joblib**

Save and load your wrapped estimators for production deployment. This example covers three serialization scenarios: standalone wrapped estimators, complete pipelines combining wrappers with Scikit-Learn transformers, and full GridSearchCV objects including all cross-validation results. You'll verify that deserialized objects maintain identical behavior to their originals, which is critical for reproducible ML workflows and model deployment.

### [nested_wrappers.py](/examples/nested_wrappers/)

**Controlling Nested Estimator Hierarchies**

Master Scikit-Learn's double-underscore syntax for nested parameters. This example builds an ensemble regressor that contains multiple wrapped estimators, then demonstrates how to access and modify parameters at any level of the hierarchy using the `outer__inner__param` pattern. You'll see interactive parameter exploration showing how `get_params(deep=True)` expands nested structures, and how `set_params()` traverses them. Essential for meta-estimators and ensemble methods.

### [fit_context.py](/examples/fit_context/)

**Understanding the _fit_context Decorator**

Explore the decorator that powers Sklearn-Wrap's automatic instantiation and validation. This example compares manual instantiation (calling `instantiate()` yourself) against the decorator approach, showing why the decorator is preferred. You'll learn about `prefer_skip_nested_validation` and when to set it True versus False, plus see how the decorator handles partial_fit scenarios for incremental learning. Deep dive into the mechanics that make everything else work smoothly.

## Next Steps

- **[User Guide](user-guide.md)** - Deep dive into core concepts and architecture
- **[API Reference](api-reference.md)** - Complete BaseClassWrapper documentation
- **[Contributing](contributing.md)** - Add your own examples or improve existing ones
