import pytest
import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import jax


def get_data_and_gp(n_datapoints):
    x = jnp.linspace(-5., 5., n_datapoints).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key=jr.PRNGKey(123), shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)

    p = gpx.Prior(kernel=gpx.RBF())
    lik = gpx.Gaussian(num_datapoints=n_datapoints)
    post = p * lik
    return D, post




@pytest.mark.parametrize('n_datapoints, n_inducing_points', [(10, 1), (100, 10)])
def test_elbo(n_datapoints, n_inducing_points):
    D, post = get_data_and_gp(n_datapoints)
    params, trainable_status, constrainer, unconstrainer = gpx.initialise(post)
    params = gpx.transform(params, unconstrainer)

    inducing_inputs = jnp.linspace(-5., 5., n_inducing_points).reshape(-1, 1)
    q = gpx.VariationalGaussian(inducing_inputs=inducing_inputs)
    svgp = gpx.SVGP(posterior=post, variational_family=q)

    assert svgp.prior == post.prior
    assert svgp.likelihood == post.likelihood
    assert svgp.num_inducing == n_inducing_points