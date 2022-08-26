from copy import deepcopy
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax import value_and_grad
from jaxtyping import f64

from .config import get_defaults
from .gps import AbstractPosterior
from .parameters import build_trainables, trainable_params, transform
from .types import Dataset
from .utils import I
from .variational_families import (
    AbstractVariationalFamily,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
)
from .variational_inference import StochasticVI

DEFAULT_JITTER = get_defaults()["jitter"]


def natural_to_expectation(params: Dict, jitter: float = DEFAULT_JITTER) -> Dict:
    """
    Translate natural parameters to expectation parameters.

    In particular, in terms of the Gaussian mean μ and covariance matrix μ for the Gaussian variational family,

        - the natural parameteristaion is θ = (S⁻¹μ, -S⁻¹/2)
        - the expectation parameters are  η = (μ, S + μ μᵀ).

    This function solves these eqautions in terms of μ and S to convert θ to η.

    Writing θ = (θ₁, θ₂), we have that S⁻¹ = -2θ₂ . Taking the cholesky decomposition of the inverse covariance,
    S⁻¹ = LLᵀ and defining C = L⁻¹, we have S = CᵀC and μ = Sθ₁ = CᵀC θ₁.

    Now from here, using μ and S found from θ, we compute η as η₁ = μ, and  η₂ = S + μ μᵀ.

    Args:
        params: A dictionary of variational Gaussian parameters under the natural parameterisation.
        jitter (float): A small value to prevent numerical instability.
    Returns:
        Dict: A dictionary of Gaussian moments under the expectation parameterisation.
    """

    natural_matrix = params["variational_family"]["moments"]["natural_matrix"]
    natural_vector = params["variational_family"]["moments"]["natural_vector"]
    m = natural_vector.shape[0]

    # S⁻¹ = -2θ₂
    S_inv = -2 * natural_matrix
    S_inv += I(m) * jitter

    # S⁻¹ = LLᵀ
    L = jnp.linalg.cholesky(S_inv)

    # C = L⁻¹I
    C = jsp.linalg.solve_triangular(L, I(m), lower=True)

    # S = CᵀC
    S = jnp.matmul(C.T, C)

    # μ = Sθ₁
    mu = jnp.matmul(S, natural_vector)

    # η₁ = μ
    expectation_vector = mu

    # η₂ = S + μ μᵀ
    expectation_matrix = S + jnp.matmul(mu, mu.T)

    params["variational_family"]["moments"] = {
        "expectation_vector": expectation_vector,
        "expectation_matrix": expectation_matrix,
    }

    return params


def _expectation_elbo(
    posterior: AbstractPosterior,
    variational_family: AbstractVariationalFamily,
    train_data: Dataset,
) -> Callable[[Dict, Dataset], float]:
    """
    Construct evidence lower bound (ELBO) for variational Gaussian under the expectation parameterisation.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
    Returns:
        Callable: A function that computes the ELBO.
    """
    expectation_vartiational_gaussian = ExpectationVariationalGaussian(
        prior=variational_family.prior,
        inducing_inputs=variational_family.inducing_inputs,
    )
    svgp = StochasticVI(
        posterior=posterior, variational_family=expectation_vartiational_gaussian
    )

    return svgp.elbo(train_data, transformations=None, negative=True)


def _rename_expectation_to_natural(params: Dict) -> Dict:
    """This function renames the gradient components (that have expectation parameterisation keys) to match the natural parameterisation pytree."""
    params["variational_family"]["moments"] = {
        "natural_vector": params["variational_family"]["moments"]["expectation_vector"],
        "natural_matrix": params["variational_family"]["moments"]["expectation_matrix"],
    }

    return params


def _rename_natural_to_expectation(params: Dict) -> Dict:
    """This function renames the gradient components (that have natural parameterisation keys) to match the expectation parameterisation pytree."""
    params["variational_family"]["moments"] = {
        "expectation_vector": params["variational_family"]["moments"]["natural_vector"],
        "expectation_matrix": params["variational_family"]["moments"]["natural_matrix"],
    }

    return params


def natural_gradients(
    stochastic_vi: StochasticVI,
    train_data: Dataset,
    transformations: Dict,
    trainables: Dict,
) -> Tuple[Callable[[Dict, Dataset], Dict]]:
    """
    Computes the gradient with respect to the natural parameters. Currently only implemented for the natural variational Gaussian family.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
        train_data: A Dataset.
        transformations: A dictionary of transformations.
    Returns:
        Tuple[Callable[[Dict, Dataset], Dict]]: Functions that compute natural gradients and hyperparameter gradients respectively.
    """
    posterior = stochastic_vi.posterior
    variational_family = stochastic_vi.variational_family

    # The ELBO under the user chosen parameterisation xi.
    xi_elbo = stochastic_vi.elbo(train_data, transformations, negative=True)

    # The ELBO under the expectation parameterisation, L(η).
    expectation_elbo = _expectation_elbo(posterior, variational_family, train_data)

    # Stop nonment params:
    expectation_trainables = _rename_natural_to_expectation(deepcopy(trainables))
    moment_trainables = build_trainables(expectation_trainables, False)
    moment_trainables["variational_family"]["moments"] = expectation_trainables[
        "variational_family"
    ]["moments"]

    # Stop moment params:
    hyper_trainables = deepcopy(trainables)
    hyper_trainables["variational_family"]["moments"] = build_trainables(
        trainables["variational_family"]["moments"], False
    )

    if isinstance(variational_family, NaturalVariationalGaussian):

        def nat_grads_fn(params: Dict, batch: Dataset) -> Dict:
            """
            Computes the natural gradients of the ELBO.
            Args:
                params: A dictionary of parameters.
                trainables: A dictionary of trainables.
                batch: A Dataset.
            Returns:
                Dict: A dictionary of natural gradients.
            """
            # Transform parameters to constrained space.
            params = transform(params, transformations)

            # Convert natural parameterisation θ to the expectation parametersation η.
            expectation_params = natural_to_expectation(params)

            # Compute gradient ∂L/∂η:
            def loss_fn(params: Dict, batch: Dataset) -> f64["1"]:
                # Stop gradients for non-trainable and non-moment parameters.
                params = trainable_params(params, moment_trainables)

                return expectation_elbo(params, batch)

            value, dL_dexp = value_and_grad(loss_fn)(expectation_params, batch)

            nat_grad = _rename_expectation_to_natural(dL_dexp)

            return value, nat_grad

    else:
        raise NotImplementedError

    def hyper_grads_fn(params: Dict, batch: Dataset) -> Dict:
        """
        Computes the hyperparameter gradients of the ELBO.
        Args:
            params: A dictionary of parameters.
            trainables: A dictionary of trainables.
            batch: A Dataset.
        Returns:
            Dict: A dictionary of hyperparameter gradients.
        """

        def loss_fn(params: Dict, batch: Dataset) -> f64["1"]:
            # Stop gradients for non-trainable and moment parameters.
            params = trainable_params(params, hyper_trainables)

            return xi_elbo(params, batch)

        value, dL_dhyper = value_and_grad(loss_fn)(params, batch)

        return value, dL_dhyper

    return nat_grads_fn, hyper_grads_fn
