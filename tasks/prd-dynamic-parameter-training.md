
# PRD: Dynamic Parameter Training with NNX Filters

## 1. Introduction/Overview

This document outlines the requirements for a new feature that refactors how model parameters are trained in GPJax. Currently, a parameter's trainability is a static property of its type (e.g., `PositiveReal` is trainable, `Static` is not). This is inflexible and requires users to redefine their models to control which parameters are trained.

This feature will introduce a dynamic, filter-based approach to parameter training, aligning GPJax with the idiomatic patterns of the Flax NNX library. Users will be able to select which parameters to train at runtime via a filter passed to the `fit` function, providing greater flexibility and control over the optimization process.

## 2. Goals

*   **Increase Flexibility:** Allow users to dynamically select which parameters to train without changing their model definitions.
*   **Improve API Design:** Provide a more powerful and intuitive API for model training.
*   **Align with NNX:** Adopt the idiomatic use of `nnx.Filter` for selecting parts of a model state, making the library more familiar to Flax/NNX users.
*   **Simplify Model Definitions:** Remove the need for the `Static` parameter type, reducing boilerplate and simplifying the code required to define models.
*   **Empower Researchers:** Give researchers complete, granular control over what parameters are trained and when.

## 3. User Stories

*   **As a researcher, I want to select which parameters to train at runtime, so that I can conduct experiments without redefining my model.**
*   **As a practitioner, I want to freeze the kernel's variance while training the lengthscale, so I can see how the model behaves with a fixed variance.**
*   **As a researcher, I want to freeze my kernel parameters so I can train only the variational parameters of my model.**
*   **As a researcher, I want complete control over what parameters are trained and when, to enable complex training workflows.**

## 4. Functional Requirements

1.  The `fit` function in `gpjax/fit.py` **must** be modified to accept an optional `trainable` argument.
2.  The `trainable` argument **must** be an `nnx.Filter`. Its default value **must** be `nnx.Param`, ensuring that all model parameters are trained if the argument is not provided.
3.  When a `trainable` filter is passed to the `fit` function, the optimizer **must** only update the parameters that match the filter.
4.  The `gpjax.parameters.Static` type **must** be completely removed from the codebase.
5.  The `Constant` mean function in `gpjax/mean_functions.py` **must** be refactored. Its `__init__` method should accept either a `Parameter` object (to create a trainable constant) or a raw float/array (to create a fixed, non-trainable constant).
6.  The `Zero` mean function **must** be updated to be a simple subclass of `Constant`, initialized with a fixed, non-trainable value of `0.0`.
7.  The value of the `Zero` mean function **must not** be trained, regardless of any filter passed to the `fit` function.
8.  A thorough search of the codebase **must** be conducted to identify and refactor any other components that rely on the `Static` parameter type, ensuring they are compatible with the new filtering mechanism.

## 5. Non-Goals (Out of Scope)

*   This feature will **not** introduce a graphical user interface (GUI) or any other interactive method for selecting parameters.
*   This feature will **not** change the underlying optimization algorithms (e.g., `optax` optimizers).
*   This feature will **not** change how parameter constraints (e.g., positivity) are handled via bijectors.

## 6. Design Considerations

This is a code-level API change and does not have UI/UX design considerations.

## 7. Technical Considerations

*   The implementation will primarily affect `gpjax/fit.py`, `gpjax/parameters.py`, and `gpjax/mean_functions.py`.
*   The core logic in `fit.py` will change from `nnx.split(model, Parameter, ...)` to `nnx.split(model, trainable_filter)`.
*   The `Constant` mean function's `__call__` method will need to handle accessing the value from both `Parameter` objects (`.value`) and raw JAX arrays.
*   Existing tests will need to be updated to reflect the removal of `Static` and the new `fit` function signature.
*   New tests should be written to verify the filtering logic and the correct behavior of the refactored `Constant` and `Zero` mean functions.

## 8. Success Metrics

*   Successful implementation of all acceptance criteria outlined by the user.
*   Positive feedback from the GPJax community regarding the new, more flexible API.
*   A measurable reduction in the lines of code required for common user tasks, such as defining a model with a mix of trained and frozen parameters.

## 9. Open Questions

*   Are there any other components in the GPJax library, besides `Constant` and `Zero` mean functions, that use a "statically frozen" parameter pattern and need to be refactored?
