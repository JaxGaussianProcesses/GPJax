from copy import deepcopy
from this import d
import typing as tp
import jax.numpy as jnp
import jax.scipy as jsp
import distrax as dx
from jax import lax, value_and_grad

from .config import get_defaults
from .variational_families import AbstractVariationalFamily, ExpectationVariationalGaussian, NaturalVariationalGaussian
from .variational_inference import StochasticVI
from .utils import I
from .gps import AbstractPosterior
from .types import Dataset, Array
from .parameters import build_identity, transform, build_trainables_false, build_trainables_true, trainable_params

DEFAULT_JITTER = get_defaults()["jitter"]


def natural_to_expectation(natural_moments: dict, jitter: float = DEFAULT_JITTER) -> dict:
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
            ) -> tp.Callable[[dict, Dataset], float]:
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


def _stop_gradients_nonmoments(params: tp.Dict) -> tp.Dict:
    """
    Stops gradients for non-moment parameters.
    Args:
        params: A dictionary of parameters.
    Returns:
        tp.Dict: A dictionary of parameters with stopped gradients.
    """
    trainables = build_trainables_false(params)
    moment_trainables = build_trainables_true(params["variational_family"]["moments"])
    trainables["variational_family"]["moments"] = moment_trainables
    params = trainable_params(params, trainables)
    return params

def _stop_gradients_moments(params: tp.Dict) -> tp.Dict:
    """
    Stops gradients for moment parameters.
    Args:
        params: A dictionary of parameters.
    Returns:
        tp.Dict: A dictionary of parameters with stopped gradients.
    """
    trainables = build_trainables_true(params)
    moment_trainables = build_trainables_false(params["variational_family"]["moments"])
    trainables["variational_family"]["moments"] = moment_trainables
    params = trainable_params(params, trainables)
    return params


def natural_gradients(
    stochastic_vi: StochasticVI,
    train_data: Dataset,
    transformations: dict,
    #bijector = tp.Optional[dx.Bijector] = Identity, #bijector: A bijector to convert between the user chosen parameterisation and the natural parameters.
 ) -> tp.Tuple[tp.Callable[[dict, Dataset], dict]]:
    """
    Computes natural gradients for variational Gaussian.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
        train_data: A Dataset.
        transformations: A dictionary of transformations.
    Returns:
        Tuple[tp.Callable[[dict, Dataset], dict]]: Functions that compute natural gradients and hyperparameter gradients respectively.
    """
    posterior = stochastic_vi.posterior
    variational_family = stochastic_vi.variational_family

    # The ELBO under the user chosen parameterisation xi.
    xi_elbo = stochastic_vi.elbo(train_data, transformations)
    
    # The ELBO under the expectation parameterisation, L(η).
    expectation_elbo = _expectation_elbo(posterior, variational_family, train_data)

    if isinstance(variational_family, NaturalVariationalGaussian):
        def nat_grads_fn(params: dict, trainables: dict, batch: Dataset) -> dict:
            """
            Computes the natural gradients of the ELBO.
            Args:
                params: A dictionary of parameters.
                trainables: A dictionary of trainables. 
                batch: A Dataset.
            Returns:
                dict: A dictionary of natural gradients.
            """
            # Transform parameters to constrained space.
            params = transform(params, transformations)
            
            # Get natural moments θ.
            natural_moments = params["variational_family"]["moments"]

            # Get expectation moments η.
            expectation_moments = natural_to_expectation(natural_moments)

            # Full params with expectation moments.
            expectation_params = deepcopy(params)
            expectation_params["variational_family"]["moments"] = expectation_moments

            # Compute gradient ∂L/∂η:
            def loss_fn(params: dict, batch: Dataset) -> Array:
                # Determine hyperparameters that should be trained.
                trains = deepcopy(trainables)
                trains["variational_family"]["moments"] = build_trainables_true(params["variational_family"]["moments"])
                params = trainable_params(params, trains) 
            
                # Stop gradients for non-moment parameters.
                params = _stop_gradients_nonmoments(params)

                return expectation_elbo(params, batch)

            value, dL_dnat = value_and_grad(loss_fn)(expectation_params, batch)

            

            # This is a renaming of the gradient components to match the natural parameterisation pytree.
            nat_grad = dL_dnat 
            nat_grad["variational_family"]["moments"] = {"natural_vector": dL_dnat["variational_family"]["moments"]["expectation_vector"],
            "natural_matrix": dL_dnat["variational_family"]["moments"]["expectation_matrix"]}

            return value, nat_grad

    else:
        #To Do: (DD) add general parameterisation case. 
        raise NotImplementedError

        # BELOW is (almost working) PSUEDO CODE of what this will look like.
        
        # def nat_grads_fn(params: dict, batch: Dataset) -> dict:
        #     # Transform parameters to constrained space.
        #     params = transform(params, transformations)

        #     # Stop gradients for non-moment parameters.
        #     params = _stop_gradients_nonmoments(params)

        #     # Get natural moments θ.
        #     natural_moments = bijector.inverse(params["variational_family"]["moments"])

        #     # Get expectation moments η.
        #     expectation_moments = natural_to_expectation(natural_moments)

        #     # Gradient function ∂ξ/∂θ:
        #     #### NEED TO STOP GRADIENTS FOR NON MOMENTS HERE!####
        #     dxi_dnat = jacobian(nat_to_moments.forward)(natural_moments)

        #     # Full params with expectation moments.
        #     expectation_params = deepcopy(params)
        #     expectation_params["variational_family"]["moments"] = expectation_moments

        #     # Compute gradient ∂L/∂η:
        #     def loss_fn(params: dict, batch: Dataset) -> Array:
        #       # Determine hyperparameters that should be trained.
        #       params = trainable_params(params, trainables)

        #       # Stop gradients for non-moment parameters.
        #       params = _stop_gradients_nonmoments(params)

        #       return expectation_elbo(expectation_params, batch)

        #     value, dL_dnat = value_and_grad(loss_fn)(expectation_params, batch)

        #     # ∂ξ/∂θ ∂L/∂η
        #     nat_grads = jax.tree_multimap(lambda x, y: jnp.matmul(x.T, y), dxi_dnat, dL_dnat)

        #     return value, nat_grads

    def hyper_grads_fn(params: dict,  trainables: dict, batch: Dataset) -> dict:
        """
        Computes the hyperparameter gradients of the ELBO.
        Args:
            params: A dictionary of parameters.
            trainables: A dictionary of trainables. 
            batch: A Dataset.
        Returns:
            dict: A dictionary of hyperparameter gradients.
        """
        # Transform parameters to constrained space.
        params = transform(params, transformations)

        def loss_fn(params: dict, batch: Dataset) -> Array:
            # Determine hyperparameters that should be trained.
            params = trainable_params(params, trainables)

            # Stop gradients for the moment parameters.
            params = _stop_gradients_moments(params)

            return xi_elbo(params, batch)

        value, dL_dhyper  =  value_and_grad(loss_fn)(params, batch)

        return  value, dL_dhyper

    return nat_grads_fn, hyper_grads_fn