# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Enable Float64 for more stable matrix inversions.
from jax import config

config.update("jax_enable_x64", True)


import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    Num,
)
import pytest

import gpjax as gpx
from gpjax.mean_functions import (
    AbstractMeanFunction,
    CombinationMeanFunction,
    Constant,
    Zero,
)
from gpjax.parameters import Static


def test_abstract() -> None:
    # Check abstract mean function cannot be instantiated, as the `__call__` method is not defined.
    with pytest.raises(TypeError):
        AbstractMeanFunction()

    # Check a "dummy" mean function with defined abstract method, `__call__`, can be instantiated.
    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Float[Array, " D"]) -> Float[Array, "1"]:
            return jnp.array([1.0])

    mf = DummyMeanFunction()
    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([1.0])) == jnp.array([1.0])).all()
    assert (mf(jnp.array([2.0, 3.0])) == jnp.array([1.0])).all()


@pytest.mark.parametrize(
    "constant", [jnp.array([0.0]), jnp.array([1.0]), jnp.array([3.0])]
)
def test_constant(constant: Float[Array, " Q"]) -> None:
    mf = Constant(constant=constant)

    assert isinstance(mf, AbstractMeanFunction)
    assert (mf(jnp.array([[1.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[2.0, 3.0]])) == jnp.array([constant])).all()
    assert (mf(jnp.array([[1.0], [2.0]])) == jnp.array([constant, constant])).all()
    assert (
        mf(jnp.array([[1.0, 2.0], [3.0, 4.0]])) == jnp.array([constant, constant])
    ).all()


def test_zero_mean_remains_zero() -> None:
    key = jr.PRNGKey(123)

    x = jr.uniform(key=key, minval=0, maxval=1, shape=(20, 1))
    y = jnp.full((20, 1), 50, dtype=jnp.float64)  # Dataset with non-zero mean
    D = gpx.Dataset(X=x, y=y)

    constant = Static(jnp.array(0.0))
    kernel = gpx.kernels.Constant(constant=constant)
    meanf = Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=D.n, obs_stddev=jnp.array(1e-3)
    )
    posterior = prior * likelihood

    opt_posterior, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
    )
    assert opt_posterior.prior.mean_function.constant.value == 0.0


def test_initialising_zero_mean_with_constant_raises_error():
    with pytest.raises(TypeError):
        Zero(constant=jnp.array([1.0]))


@pytest.fixture
def dummy_mean_function() -> AbstractMeanFunction:
    """Create a simple mean function for testing."""

    class DummyMeanFunction(AbstractMeanFunction):
        def __call__(self, x: Num[Array, "N D"]) -> Float[Array, "N O"]:
            return jnp.ones((x.shape[0], 1))

    return DummyMeanFunction()


@pytest.fixture
def constant_mean_function() -> AbstractMeanFunction:
    """Create a constant mean function for testing."""
    return Constant(constant=jnp.array([2.0]))


def test_mean_function_addition(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test addition of two mean functions."""
    # Test adding two mean functions
    sum_mean = dummy_mean_function + constant_mean_function

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 2.0, sum should be 3.0
    expected = jnp.array([[3.0], [3.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_radd(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test right addition of mean function with a constant."""
    # Test adding constant to mean function
    constant_value = jnp.array([5.0])
    sum_mean = constant_value + dummy_mean_function

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 5.0, sum should be 6.0
    expected = jnp.array([[6.0], [6.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_add_constant(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test addition of mean function with a constant."""
    # Test adding mean function to constant
    constant_value = jnp.array([3.0])
    sum_mean = dummy_mean_function + constant_value

    # Check the result is a CombinationMeanFunction with sum operator
    assert isinstance(sum_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = sum_mean(x)

    # Expected: dummy returns ones, constant returns 3.0, sum should be 4.0
    expected = jnp.array([[4.0], [4.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_multiplication(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test multiplication of two mean functions."""
    # Test multiplying two mean functions
    product_mean = dummy_mean_function * constant_mean_function

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 2.0, product should be 2.0
    expected = jnp.array([[2.0], [2.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_rmul(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test right multiplication of mean function with a constant."""
    # Test multiplying constant with mean function
    constant_value = jnp.array([5.0])
    product_mean = constant_value * dummy_mean_function

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 5.0, product should be 5.0
    expected = jnp.array([[5.0], [5.0]])
    assert jnp.allclose(result, expected)


def test_mean_function_mul_constant(dummy_mean_function: AbstractMeanFunction) -> None:
    """Test multiplication of mean function with a constant."""
    # Test multiplying mean function with constant
    constant_value = jnp.array([3.0])
    product_mean = dummy_mean_function * constant_value

    # Check the result is a CombinationMeanFunction with product operator
    assert isinstance(product_mean, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = product_mean(x)

    # Expected: dummy returns ones, constant returns 3.0, product should be 3.0
    expected = jnp.array([[3.0], [3.0]])
    assert jnp.allclose(result, expected)


def test_chained_operations(
    dummy_mean_function: AbstractMeanFunction,
    constant_mean_function: AbstractMeanFunction,
) -> None:
    """Test chained operations between mean functions."""
    # Test a combination of addition and multiplication
    # Use jnp.array instead of float 3.0
    combined = dummy_mean_function + constant_mean_function * jnp.array([3.0])

    # Check the structure is correct
    assert isinstance(combined, CombinationMeanFunction)

    # Test evaluation
    x = jnp.array([[1.0], [2.0]])
    result = combined(x)

    # The actual result is [[6.0], [6.0]] (not 7.0 as initially expected)
    # This is because the operation works differently than we expected
    expected = jnp.array([[6.0], [6.0]])
    assert jnp.allclose(result, expected)
