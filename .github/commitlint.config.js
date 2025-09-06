module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'build',    // Changes that affect the build system or external dependencies
        'chore',    // Other changes that don't modify src or test files
        'ci',       // Changes to CI configuration files and scripts
        'docs',     // Documentation only changes
        'feat',     // A new feature
        'fix',      // A bug fix
        'perf',     // A code change that improves performance
        'refactor', // A code change that neither fixes a bug nor adds a feature
        'revert',   // Reverts a previous commit
        'style',    // Changes that do not affect the meaning of the code
        'test',     // Adding missing tests or correcting existing tests
      ],
    ],
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],
    'scope-enum': [
      2,
      'always',
      [
        // Core components
        'kernels',     // Kernel implementations and computations
        'gps',         // Gaussian process models and inference
        'likelihoods', // Likelihood functions
        'mean-functions', // Mean function implementations
        'variational', // Variational inference methods
        'objectives',  // Optimization objectives
        'parameters',  // Parameter management and transformations
        'utils',       // Utility functions
        'linalg',      // Linear algebra utilities
        'scan',        // Scan operations
        'dataset',     // Dataset handling
        'fit',         // Model fitting and optimization
        
        // Testing and validation
        'tests',       // Test improvements
        'examples',    // Example notebooks and scripts
        'benchmarks',  // Performance benchmarks
        'validation',  // Model validation
        
        // Infrastructure
        'ci',          // Continuous integration
        'docs',        // Documentation
        'deps',        // Dependency updates
        'release',     // Release related changes
        'security',    // Security improvements
        'performance', // Performance improvements
        
        // Development
        'dev',         // Development tools
        'typing',      // Type annotations
        'format',      // Code formatting
        'lint',        // Linting improvements
      ],
    ],
    'scope-case': [2, 'always', 'lower-case'],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'header-max-length': [2, 'always', 72],
    'body-leading-blank': [2, 'always'],
    'footer-leading-blank': [2, 'always'],
  },
};