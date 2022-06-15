import typing as tp
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jacobian
import distrax as dx
from jax import lax

from .config import get_defaults
from .variational_families import AbstractVariationalFamily, ExpectationVariationalGaussian
from .variational_inference import StochasticVI
from .utils import I
from .gps import AbstractPosterior
from .types import Dataset
from .parameters import build_identity, transform

DEFAULT_JITTER = get_defaults()["jitter"]


def natural_to_expectation(natural_moments: dict, jitter: float = DEFAULT_JITTER):
    """
    Converts natural parameters to expectation parameters.
    Args:
        natural_moments: A dictionary of natural parameters.
        jitter (float): A small value to prevent numerical instability.
    Returns:
        tp.Dict: A dictionary of Gaussian moments under the expectation parameterisation.
    """
    
    natural_matrix = natural_moments["natural_matrix"]
    natural_vector = natural_moments["natural_vector"]
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


def _expectation_elbo(posterior: AbstractPosterior,
            variational_family: AbstractVariationalFamily,
            train_data: Dataset,
            ):
    """
    Construct evidence lower bound (ELBO) for varational Gaussian under the expectation parameterisation.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
    Returns:
        Callable: A function that computes ELBO.
    """
    evg = ExpectationVariationalGaussian(prior= variational_family.prior, 
                                        inducing_inputs = variational_family.inducing_inputs,
                                        )
    svgp = StochasticVI(posterior=posterior, variational_family=evg)

    return svgp.elbo(train_data, build_identity(svgp.params))


def natural_gradients(
    posterior: AbstractPosterior,
    variational_family: AbstractVariationalFamily,
    train_data: Dataset,
    params: dict,
    transformations: dict,
    nat_to_moments: dx.Bijector,
    batch,
) -> dict:
    """
    Computes natural gradients for a variational family.
    Args:
        posterior (AbstractPosterior): An instance of AbstractPosterior.
        variational_family(AbstractVariationalFamily): An instance of AbstractVariationalFamily.
        train_data (Dataset): Training Dataset.
        params (tp.Dict): A dictionary of model parameters.
        transformations (tp.Dict): A dictionary of parameter transformations.
        nat_to_moments (dx.Bijector): A bijector between natural and the chosen parameterisations of the Gaussian variational moments.
    Returns:
        tp.Dict: Dictionary of natural gradients.
    """
    # Transform the parameters.
    params = transform(params, transformations)

    # Get moments and stop gradients for non-moment parameters.
    moments = params["variational_family"]["moments"]

    other_var_params = {k:v for k,v in params["variational_family"].items() if k!="moments"}
    other_params = lax.stop_gradient({**{k:v for k,v in params.items() if k!="variational_family"}, **other_var_params})
    
    # Convert moments to natural parameterisation.
    natural_moments = nat_to_moments.inverse(moments)

    # Gradient function ∂ξ/∂θ:
    dxi_dnat = jacobian(nat_to_moments.forward)(natural_moments)

    # Convert natural moments to expectation moments.
    expectation_moments = natural_to_expectation(natural_moments)

    # Create dictionary of all parameters for the ELBO under the expectation parameterisation.
    expectation_params = other_params
    expectation_params["variational_family"]["moments"] = expectation_moments

    # Compute ELBO.
    expectation_elbo = _expectation_elbo(posterior, variational_family, train_data)(expectation_params, batch) 

    # Compute gradient ∂L/∂η:
    dL_dnat = jacobian(expectation_elbo)(expectation_params)
    
    # Compute natural gradient:
    nat_grads = jnp.matmul(dxi_dnat, dL_dnat.T) #<---- PSUEDO CODE - TO DO - Pytree operations needed here.
    
    return nat_grads
