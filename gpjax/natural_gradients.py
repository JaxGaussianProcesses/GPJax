from cmath import exp
import typing as tp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jacobian

from .config import get_defaults
from .variational_families import AbstractVariationalFamily, ExpectationVariationalGaussian
from .variational_inference import StochasticVI
from .utils import I
from .gps import AbstractPosterior
from .types import Dataset
from .parameters import build_identity, transform

DEFAULT_JITTER = get_defaults()["jitter"]

# CURRENTLY THIS FILE IS A FIRST SKETCH OF NATURAL GRADIENTS in GPJax.

# Below is correct, but it might be better to pass in params (i.e., all svgp params) and return a dictionary that gives svgp params
def natural_to_expectation(natural_params: dict, jitter: float = DEFAULT_JITTER):
    """
    Converts natural parameters to expectation parameters.
    Args:
        natural_params: A dictionary of natural parameters.
        jitter: A small value to prevent numerical instability.
    Returns:
        A dictionary of expectation parameters.
    """
    
    natural_matrix = natural_params["natural_matrix"]
    natural_vector = natural_params["natural_vector"]
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
    
    # η₂ = S + η₁ η₁ᵀ
    expectation_matrix = S + jnp.matmul(mu, mu.T)
    
    return {"expectation_vector": expectation_vector, "expectation_matrix": expectation_matrix}


# This is a function that you create before training. This can be used to get the elbo for the nexpectation parameterisation.
# Here it is assumed that the parameters have already been transformed prior to being passed to the returned function.
def get_expectation_elbo(posterior: AbstractPosterior,
            variational_family: AbstractVariationalFamily,
            train_data: Dataset,
            ):
    """
    Computes evidence lower bound (ELBO) for the expectation parameterisation.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
    Returns:
        Callable: A function that computes ELBO.
    """
    q = variational_family
    expectaction_q = ExpectationVariationalGaussian(prior=q.prior, inducing_inputs = q.inducing_inputs)
    svgp = StochasticVI(posterior=posterior, variational_family=expectaction_q)
    transformations = build_identity(svgp.params)

    return svgp.elbo(train_data, transformations)


def natural_gradients(params: dict,
            transformations: dict,
            expectation_elbo: tp.Callable,
            nat_to_xi: tp.Callable, 
            xi_to_nat: tp.Callable, 
            batch,
) -> dict:
    """
    Computes natural gradients for a variational family.
    Args:
        params (tp.Dict): A dictionary of parameters.
        variational_family: A variational family.
        nat_to_xi: A function that converts natural parameters to variational parameters xi.
        xi_to_nat: A function that converts variational parameters xi to natural parameters.
        transformations (tp.Dict): A dictionary of transformations.
    Returns:
        tp.Dict: Dictionary of natural gradients.
    """
    # Transform the parameters.
    params = transform(params, transformations)

    # Need to stop gradients for hyperparameters.

    natural_params = xi_to_nat(params)

    # Gradient function ∂ξ/∂θ:
    dxi_dnat = jacobian(nat_to_xi)(natural_params)

    expectation_params = natural_to_expectation(natural_params)
    expectation_elbo = expectation_elbo(expectation_params, batch) 

    # Compute gradient ∂L/∂η:
    dL_dnat = jacobian(expectation_elbo)(expectation_params)
    
    # Compute natural gradient:
    nat_grads = jnp.matmul(dxi_dnat, dL_dnat.T) #<- Some pytree operations are needed here.
    
    return nat_grads
