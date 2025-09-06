## 

Success criteria: `uv run poe all-tests` passes. Nothing else will be accepted.

## Relevant Files

- `gpjax/parameters.py`: Contains the core parameter definitions. Will be modified to remove the `Static` type.
- `tests/test_parameters.py`: Unit tests for `parameters.py`. Will need updates to remove tests for `Static`.
- `gpjax/mean_functions.py`: Contains the `Constant` and `Zero` mean function logic that will be refactored.
- `tests/test_mean_functions.py`: Unit tests for `mean_functions.py`. Will need to be updated to test the new behavior.
- `gpjax/fit.py`: Contains the core `fit` function whose signature and logic will be updated.
- `tests/test_fit.py`: Unit tests for `fit.py`. New tests will be added here to verify the filtering functionality.
- `examples/`: Various files in this directory may need to be updated to reflect the new API for freezing parameters.

### Notes

- Ensure that `uv run` is used to prepend commands.
- Unit tests are located in the top-level `tests/` directory, which mirrors the structure of the `gpjax/` source code.
- Use `pytest` to run tests. Running `pytest` from the root directory will automatically discover and run all test files. To run a specific test file, use `pytest tests/path/to/test_file.py`.

## Tasks

- [ ] 1.0 Refactor core parameter types and remove `Static`
  - [ ] 1.1 In `gpjax/parameters.py`, delete the `Static` class definition.
  - [ ] 1.2 Search the entire codebase for any remaining usages of the `Static` type and replace them. For most cases, like the `Zero` mean function, this will involve using a raw float value instead of a `Static` object.
  - [ ] 1.3 Update any import statements that specifically reference `Static`.

- [ ] 2.0 Refactor `Constant` and `Zero` mean functions
  - [ ] 2.1 In `gpjax/mean_functions.py`, modify the `Constant` class `__init__` method to accept either a `Parameter` object or a raw float/array.
  - [ ] 2.2 Add logic to `Constant.__init__` to store the input as `self.constant`. If the input is a `Parameter`, store it directly. If it's a raw value, store it as a `jnp.array`.
  - [ ] 2.3 Update the `Constant.__call__` method to handle both cases: access `.value` if `self.constant` is a `Parameter`, or use the raw array directly otherwise.
  - [ ] 2.4 Modify the `Zero` class `__init__` method to simply call `super().__init__(constant=0.0)`.

- [ ] 3.0 Update the `fit` function to use `nnx.Filter`
  - [ ] 3.1 In `gpjax/fit.py`, modify the signature of the `fit` function (and its variants `fit_lbfgs`, `fit_scipy`) to accept a new optional argument `trainable: nnx.Filter = nnx.Param`.
  - [ ] 3.2 In the `fit` function, replace the logic that splits parameters using `nnx.split(model, Parameter, ...)` with the new filter: `params, static_state = nnx.split(model, trainable)`.
  - [ ] 3.3 Ensure that the optimizer is initialized with, and updates are applied to, the `params` `State` object returned by the new split logic.
  - [ ] 3.4 After the optimization loop, merge the updated `params` back into the main `model` object before returning it.

- [ ] 4.0 Create new tests for the filtering functionality
  - [ ] 4.1 In `tests/test_fit.py`, write a new test using a model with a kernel (e.g., `RBF`).
  - [ ] 4.2 Call `fit` with a filter that freezes the kernel's variance. Use `nnx.filters.Not(nnx.filters.PathContains("variance"))`.
  - [ ] 4.3 Assert that the variance parameter's value does not change after training, while the lengthscale parameter's value does.
  - [ ] 4.4 Add a test to confirm that the value of a `Zero` mean function's constant is not trained, even with the default `trainable=nnx.Param` filter.
  - [ ] 4.5 In `tests/test_mean_functions.py`, update the tests for `Constant` to verify it works correctly with both fixed raw values and trainable `Parameter` objects.
  - [ ] 4.6 Add a test that uses `nnx.filters.OfType(PositiveReal)` to demonstrate type-based filtering.

- [ ] 5.0 Update documentation and examples to reflect the new API
  - [ ] 5.1 Review all files in the `examples/` directory for any usage of the `Static` parameter.
  - [ ] 5.2 Update any affected examples to use the new filtering mechanism. Showcase using `nnx.filters.PathContains` and `nnx.filters.Not` to select parameters by name.
  - [ ] 5.3 Add a new, simple example specifically demonstrating how to freeze a kernel parameter (e.g., variance or lengthscale) during training.
  - [ ] 5.4 Update the docstrings for `fit`, `Constant`, and `Zero` to explain the new `trainable` filter and the updated behavior of `Constant`.
  - [ ] 5.5 Add a small section to the documentation (e.g., in a "Training Models" guide) that explicitly explains how to use `nnx.filters.PathContains`, `nnx.filters.OfType`, and `nnx.filters.WithTag` to build powerful training filters.
