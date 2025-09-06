⏺ GPJax Refactoring Implementation Plan

  For Junior Software Engineers

  🎯 Overview & Success Criteria

  CRITICAL: Follow Test-Driven Development (TDD) throughout. Your success criteria for each task is:
  uv run poe all-tests
  This command must pass after each completed task. If it fails, you must fix the issues before proceeding to the next task.

  Development Workflow:
  1. Run uv run poe all-tests to establish baseline (ensure it passes)
  2. Write/modify tests first (where applicable)
  3. Implement the change
  4. Run uv run poe all-tests to verify success
  5. Commit your changes with descriptive messages

  ---
  📋 Task 1: Fix __init__.py exports (High Priority)

  Estimated time: 2-3 hours

  1.1 Audit current exports

  - Open gpjax/__init__.py
  - Create a list of all items in __all__
  - For each item, verify it's actually imported/defined in the file
  - Document which items are missing/broken

  1.2 Fix missing exports

  - Remove undefined items from __all__ list (like "base", "Module", "param_field")
  - Add proper imports for any items that should be exported but aren't imported
  - Ensure all imports use absolute paths (e.g., from gpjax.gps import Prior)

  1.3 Test the fixes

  - Create test file tests/test_imports.py if it doesn't exist
  - Add test that imports all items in __all__:
  def test_all_exports_importable():
      import gpjax
      for item in gpjax.__all__:
          assert hasattr(gpjax, item), f"{item} not available in gpjax module"
  - Run uv run poe all-tests to verify

  1.4 Validation

  - Run python -c "import gpjax; print('All imports successful')"
  - Run uv run poe all-tests - must pass
  - Commit changes: "Fix init.py exports - remove undefined items from all"

  ---
  📋 Task 2: Create jitter utility function (High Priority)

  Estimated time: 4-6 hours

  2.1 Research existing jitter usage

  - Search codebase for pattern: grep -r "jnp.eye.*jitter" gpjax/
  - Search for pattern: grep -r "\+ jnp.eye" gpjax/
  - Document all locations and variations of jitter addition
  - Note parameter names used (jitter, self.jitter, etc.)

  2.2 Design the utility function

  - Create gpjax/linalg/utils.py (or add to existing file in linalg/)
  - Design function signature:
  def add_jitter(matrix: Matrix, jitter: float = 1e-6) -> Matrix:
      """Add jitter to diagonal of matrix for numerical stability."""
  - Consider whether to use CoLA operators or JAX directly

  2.3 Implement the utility function

  - Write the implementation with proper type annotations
  - Add comprehensive docstring with examples
  - Handle edge cases (empty matrix, non-square matrix)
  - Import necessary JAX/CoLA modules

  2.4 Write tests for utility function

  - Create tests/test_linalg_utils.py or add to existing test file
  - Test cases needed:
    - Square matrix with default jitter
    - Square matrix with custom jitter value
    - Edge case: 1x1 matrix
    - Edge case: Large matrix (performance)
    - Type checking with different input types

  2.5 Replace existing jitter patterns

  - Start with gpjax/variational_families.py
    - Import your utility function
    - Replace each + jnp.eye(shape[0]) * jitter with add_jitter(matrix, jitter)
    - Run tests after each replacement: uv run pytest tests/test_variational_families.py -v
  - Continue with gpjax/objectives.py
  - Continue with gpjax/gps.py
  - Replace all other instances found in step 2.1

  2.6 Validation

  - Run uv run poe all-tests - must pass
  - Run performance test to ensure no regression
  - Commit changes: "Add jitter utility function and replace 22+ duplicate implementations"

  ---
  📋 Task 3: Replace *args, **kwargs in abstract methods (High Priority)

  Estimated time: 6-8 hours

  3.1 Identify all abstract methods with generic signatures

  - Search for: grep -r "\*args.*\*\*kwargs" gpjax/
  - Focus on abstract methods in base classes
  - Document each method's actual usage by examining concrete implementations

  3.2 Analyze concrete implementations

  - For each abstract method found, examine all concrete implementations
  - Document the actual parameters used across implementations
  - Design proper type-annotated signatures

  3.3 Fix abstract method signatures (one file at a time)

  3.3a Start with gpjax/gps.py

  - Write tests first - create test cases that verify method signatures
  - Update abstract method signatures with proper type annotations
  - Update all concrete implementations to match
  - Run uv run pytest tests/test_gps.py -v after each change

  3.3b Continue with gpjax/likelihoods.py

  - Follow same pattern as above
  - Update abstract methods and implementations
  - Run uv run pytest tests/test_likelihoods.py -v

  3.3c Finish with gpjax/variational_families.py

  - Follow same pattern as above
  - Update abstract methods and implementations
  - Run uv run pytest tests/test_variational_families.py -v

  3.4 Update type checking

  - Ensure all new signatures use proper jaxtyping annotations
  - Add beartype checking where appropriate
  - Update docstrings to reflect new signatures

  3.5 Validation

  - Run uv run poe all-tests - must pass
  - Test with IDE to verify improved autocomplete/type checking
  - Commit changes: "Replace generic *args, **kwargs with typed signatures in abstract methods"

  ---
  📋 Task 4: Split large files into focused modules (High Priority)

  Estimated time: 8-12 hours

  4.1 Plan the split for gpjax/variational_families.py

  4.1a Analyze current structure

  - Open gpjax/variational_families.py and identify distinct classes/functionalities
  - Group related functionality (e.g., base classes, Gaussian variants, natural variants)
  - Plan directory structure (e.g., gpjax/variational_families/)

  4.1b Create module structure

  - Create gpjax/variational_families/ directory
  - Create __init__.py with proper exports
  - Plan individual files:
    - base.py - Abstract base classes
    - gaussian.py - Gaussian variational families
    - natural.py - Natural parameterization
    - whitened.py - Whitened parameterization

  4.2 Execute the split

  4.2a Move base classes first

  - Create gpjax/variational_families/base.py
  - Move AbstractVariationalFamily and related classes
  - Update imports in the new file
  - Update gpjax/variational_families/__init__.py to export these classes

  4.2b Move specific implementations

  - Create remaining files and move appropriate classes
  - Ensure each file has proper imports and exports
  - Update __init__.py to maintain backward compatibility

  4.2c Update main variational_families.py

  - Replace content with imports from submodules
  - Or replace with from .base import * style imports
  - Ensure backward compatibility

  4.3 Test the variational families split

  - Run uv run pytest tests/test_variational_families.py -v
  - Run full test suite: uv run poe all-tests
  - Fix any import issues

  4.4 Plan and execute split for gpjax/gps.py

  4.4a Analyze and plan

  - Identify distinct classes: AbstractPrior, Prior, AbstractPosterior, ConjugatePosterior, etc.
  - Plan split: gpjax/gps/base.py, gpjax/gps/priors.py, gpjax/gps/posteriors.py

  4.4b Execute the split

  - Follow same pattern as variational families
  - Create directory structure
  - Move classes to appropriate files
  - Update imports and exports
  - Maintain backward compatibility

  4.5 Test the gps split

  - Run uv run pytest tests/test_gps.py -v
  - Run full test suite: uv run poe all-tests

  4.6 Update all imports throughout codebase

  - Search for imports of split modules: grep -r "from gpjax.gps import" .
  - Update any imports that may have broken
  - Update example files if necessary

  4.7 Validation

  - Run uv run poe all-tests - must pass
  - Test imports in Python REPL to ensure backward compatibility
  - Commit changes: "Split large files: variational_families.py and gps.py into focused modules"

  ---
  📋 Task 5: Refactor complex functions with better variable names (Medium Priority)

  Estimated time: 6-8 hours

  5.1 Identify complex functions

  - Focus on gpjax/variational_families.py:198-259 (the 61-line predict method)
  - Find other functions >50 lines or with complexity issues
  - Document current variable names and their meanings

  5.2 Create mapping of cryptic variables to descriptive names

  - Lz_inv_Kzt → inducing_chol_inv_cross_cov
  - Ktz_Kzz_inv_sqrt → cross_cov_times_inducing_inv_sqrt
  - Create comprehensive mapping document

  5.3 Write tests for current behavior

  - Create comprehensive test cases that verify current function behavior
  - Test edge cases and different input combinations
  - These tests will ensure refactoring doesn't break functionality

  5.4 Refactor the predict method

  - Break into smaller helper methods (e.g., _compute_posterior_mean, _compute_posterior_variance)
  - Replace cryptic variable names with descriptive ones
  - Add intermediate validation/assertions
  - Add comprehensive docstrings

  5.5 Apply same pattern to other complex methods

  - Identify and refactor other overly complex methods
  - Focus on readability and maintainability

  5.6 Validation

  - Run specific tests: uv run pytest tests/test_variational_families.py -v
  - Run uv run poe all-tests - must pass
  - Commit changes: "Refactor complex functions with descriptive variable names and helper methods"

  ---
  📋 Task 6: Add input validation (Medium Priority)

  Estimated time: 4-6 hours

  6.1 Identify functions needing validation

  - Focus on gpjax/gps.py:225-257 and similar functions
  - Look for functions that process test_inputs or similar parameters
  - Document current lack of validation

  6.2 Design validation patterns

  - Create validation utility functions (e.g., validate_shape, validate_finite)
  - Design consistent error messages
  - Decide on validation vs assertion approach

  6.3 Write validation utilities

  - Create gpjax/utils/validation.py or add to existing utils
  - Implement common validation functions
  - Write tests for validation functions

  6.4 Add validation to key functions

  - Start with GP prediction methods
  - Add shape validation, finite value checks, etc.
  - Ensure error messages are helpful for users

  6.5 Test validation behavior

  - Write tests that verify proper error handling
  - Test with invalid inputs to ensure good error messages
  - Ensure valid inputs still work correctly

  6.6 Validation

  - Run uv run poe all-tests - must pass
  - Test with invalid inputs to verify error handling
  - Commit changes: "Add input validation to prevent cryptic runtime errors"

  ---
  📋 Task 7: Audit and optimize JAX usage patterns (Lower Priority)

  Estimated time: 6-10 hours

  7.1 Audit current CoLA usage

  - Search for to_dense() calls: grep -r "to_dense()" gpjax/
  - Document each usage and whether it's necessary
  - Identify patterns where structured operators could be maintained

  7.2 Create performance benchmarks

  - Write simple benchmarks for key operations
  - Measure current performance with dense conversions
  - These will help measure improvement

  7.3 Optimize CoLA usage patterns

  - Replace unnecessary to_dense() calls with structured operations
  - Maintain linear operator types where possible
  - Focus on high-impact areas first

  7.4 Fix inefficient matrix operations

  - Replace Identity(Kxx.shape).to_dense() * self.jitter with jnp.eye(Kxx.shape[0]) * self.jitter
  - Find similar inefficient patterns and fix them

  7.5 Test performance improvements

  - Run benchmarks to measure improvements
  - Ensure correctness is maintained
  - Document performance gains

  7.6 Validation

  - Run uv run poe all-tests - must pass
  - Run performance benchmarks to verify improvements
  - Commit changes: "Optimize JAX usage patterns - reduce unnecessary dense conversions"

  ---
  📋 Task 8: Optimize Matrix Operations (Lower Priority)

  Estimated time: 2-4 hours

  8.1 Identify inefficient patterns

  - Search for Identity(.*).to_dense() patterns
  - Find other matrix operation inefficiencies
  - Document current vs optimal implementations

  8.2 Replace with direct JAX operations

  - Replace identified patterns with efficient JAX operations
  - Ensure numerical equivalence
  - Test performance impact

  8.3 Validation

  - Run uv run poe all-tests - must pass
  - Verify performance improvements with simple benchmarks
  - Commit changes: "Replace inefficient matrix operations with direct JAX calls"

  ---
  🚀 Final Integration & Testing

  Integration Testing

  - Run complete test suite: uv run poe all-tests
  - Test example notebooks to ensure they still work
  - Run performance benchmarks to verify improvements

  Documentation Updates

  - Update any documentation affected by API changes
  - Update CHANGELOG if the project has one
  - Update docstrings that may have been affected

  Final Validation

  - uv run poe all-tests must pass
  - All examples should run without errors
  - Performance should be same or better than baseline

  ---
  ⏱️ Estimated Total Time: 40-60 hours

  Suggested ordering for maximum impact:
  1. Task 1 (quick win, fixes immediate issues)
  2. Task 2 (eliminates lots of duplication)
  3. Task 6 (improves user experience)
  4. Task 3 (improves API quality)
  5. Tasks 7-8 (performance optimizations)
  6. Task 4 (major refactoring, do last)
  7. Task 5 (polish and maintainability)

  Remember: After each task completion, run uv run poe all-tests and ensure it passes before proceeding to the next task. This is your primary success criterion.