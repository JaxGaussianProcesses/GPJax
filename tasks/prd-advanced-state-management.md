
# PRD: Advanced State Management with NNX Variables

## 1. Introduction/Overview

This document outlines the requirements for an architectural enhancement to GPJax's state management. The goal is to leverage a wider range of `flax.experimental.nnx.Variable` types to improve performance and provide richer training diagnostics. 

Specifically, this feature will introduce:
1.  **`nnx.Cache`:** To store the results of expensive but repeatable computations, such as kernel matrix inversions or Cholesky decompositions, avoiding redundant calculations.
2.  **`nnx.Intermediate`:** To provide a structured way of logging internal model values during training (e.g., individual components of an objective function) for more insightful monitoring and debugging.

This is primarily an internal architectural refactor that will provide significant, seamless benefits to users.

## 2. Goals

*   **Improve Performance:** Reduce training and inference time by caching the results of expensive matrix operations.
*   **Enhance Monitoring:** Provide users with more detailed, structured diagnostics from the training process, going beyond a single loss value.
*   **Deepen NNX Integration:** Align GPJax more closely with the full capabilities of the NNX state management system.

## 3. User Stories

*   **Caching:** "As a researcher working with a large dataset, I want the system to automatically cache kernel matrix computations within a single training step, so that my training process is faster and more efficient."
*   **Intermediate Logging:** "As a practitioner debugging a convergence issue, I want to track the log-likelihood and KL-divergence terms of my ELBO separately during training, so I can identify which component is causing the problem."

## 4. Functional Requirements

1.  **Caching Mechanism:**
    *   Model components that perform expensive, repeatable computations (e.g., `AbstractPosterior`) **must** be refactored to use `nnx.Cache` for storing results like Cholesky decompositions.
    *   A cache invalidation mechanism **must** be implemented. The training loop in `gpjax/fit.py` **must** be responsible for clearing all model caches at the beginning of each optimization step to ensure correctness.

2.  **Intermediate Value Logging:**
    *   Objective functions (e.g., `ELBO` in `gpjax/objectives.py`) **must** be refactored to store their constituent components (e.g., log-likelihood, KL-divergence) in `nnx.Intermediate` variables.
    *   The `fit` function **must** be updated to collect all `nnx.Intermediate` values from the model at each step of the optimization.
    *   The `history` object returned by the `fit` function **must** be updated to contain these collected intermediate values, providing a structured log of the training process.

## 5. Non-Goals (Out of Scope)

*   This feature will **not** require the average user to interact directly with `nnx.Cache` or `nnx.Intermediate` variables. The benefits should be delivered "under the hood" as a seamless improvement.
*   This feature will **not** introduce new optimization algorithms; it will only enhance the monitoring of existing ones.

## 6. Technical Considerations

*   The primary technical challenge is robust cache invalidation. The proposed solution is to clear caches at the start of each training step within the `fit` function. This is a simple and robust strategy that guarantees correctness.
*   The implementation will likely affect `gpjax/gps.py` (for caching), `gpjax/objectives.py` (for intermediate values), and `gpjax/fit.py` (for cache invalidation and history collection).
*   The structure of the `history` object returned by `fit` will change. This is a breaking change to the API that should be clearly documented.

## 7. Success Metrics

*   A benchmark test comparing training time before and after the caching implementation **must** show a measurable performance improvement.
*   The `history` object returned by the `fit` function **must** contain a structured log of intermediate values, not just a single loss value.
*   All existing tests **must** pass after the refactor.
