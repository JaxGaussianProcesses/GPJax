## Relevant Files

- `gpjax/gps.py`: Key posterior classes like `AbstractPosterior` will be modified to include `nnx.Cache` for matrix decompositions.
- `tests/test_gps.py`: Unit tests will be added to verify the correctness of the caching logic.
- `gpjax/objectives.py`: Objective functions like `ELBO` will be modified to log their constituent components using `nnx.Intermediate`.
- `gpjax/fit.py`: The core training loop will be updated to handle cache invalidation and to collect and return intermediate values.
- `tests/test_fit.py`: Unit tests will be added to verify that the `fit` function returns a history object with the correct intermediate values.
- `benchmarks/`: A new or existing benchmark script will be used to verify the performance improvements from caching.

### Notes

- The most critical part of this implementation is ensuring correct cache invalidation to prevent stale data. The proposed strategy is to clear all caches at the beginning of each optimization step in the `fit` function.
- The structure of the `history` object returned by `fit` will be a breaking change. This should be clearly communicated in the documentation and release notes.

## Tasks

- [ ] 1.0 Implement `nnx.Cache` for expensive computations in posterior objects
  - [ ] 1.1 In `gpjax/gps.py`, identify methods within `AbstractPosterior` that perform expensive, repeatable computations (e.g., the Cholesky decomposition of the kernel matrix).
  - [ ] 1.2 Add `nnx.Cache` attributes to the relevant posterior classes to store the results of these computations (e.g., `self.Kuu_chol: nnx.Cache[Array] = nnx.Cache(None)`).
  - [ ] 1.3 Refactor the identified methods to implement a "read-through" cache pattern: first check if the cache is populated and return the value; if not, perform the computation, store the result in the cache, and then return it.

- [ ] 2.0 Implement `nnx.Intermediate` for logging in objective functions
  - [ ] 2.1 In `gpjax/objectives.py`, identify objective functions like `ELBO` that have distinct, monitorable components (e.g., log-likelihood and KL-divergence).
  - [ ] 2.2 Within the `__call__` method of these objectives, after calculating the components, store them in `nnx.Intermediate` variables on the object (e.g., `self.log_likelihood = nnx.Intermediate(log_likelihood)`).

- [ ] 3.0 Update the `fit` function to manage caching and collect intermediate values
  - [ ] 3.1 In `gpjax/fit.py`, create a new utility function `clear_caches(model: nnx.Module)` that traverses the model's state and sets the `.value` of all `nnx.Cache` instances to `None`.
  - [ ] 3.2 In the main training loop (`scan` or `while_loop`) within the `fit` function, call `clear_caches(model)` at the beginning of each step to prevent using stale data.
  - [ ] 3.3 At the end of each training step, extract all `nnx.Intermediate` values from the updated model using `model.filter(nnx.Intermediate)`.
  - [ ] 3.4 Modify the `history` collection logic to append the structured dictionary of intermediate values from the previous step, rather than just the total loss.
  - [ ] 3.5 Update the return type hint and docstring of the `fit` function to reflect the new, richer `history` object.

- [ ] 4.0 Create new tests and benchmarks to verify performance and correctness
  - [ ] 4.1 In `tests/test_gps.py`, write a unit test to ensure the caching mechanism works. You can mock the expensive computation and assert that it is only called once when the corresponding method is called multiple times.
  - [ ] 4.2 In `tests/test_fit.py`, write a unit test to verify that the `history` object returned by `fit` now contains the expected intermediate values with the correct structure.
  - [ ] 4.3 Create a new benchmark script that trains a moderately large GP model.
  - [ ] 4.4 Run the benchmark on the codebase before and after the caching implementation to measure and assert a significant performance improvement.
