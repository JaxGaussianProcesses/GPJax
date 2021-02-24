from .gps import ConjugatePosterior
from .utils import merge_dictionaries
from .kernel import initialise
from .likelihoods import initialise
from multipledispatch import dispatch


@dispatch(ConjugatePosterior)
def initialise(gp: ConjugatePosterior):
    kernel_params = initialise(gp.prior.kernel)
    likelihood_params = initialise(gp.likelihood)
    return merge_dictionaries(kernel_params, likelihood_params)


