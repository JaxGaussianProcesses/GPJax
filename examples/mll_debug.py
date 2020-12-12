import gpflow
import numpy as np
import tensorflow_probability as tfp


if __name__=='__main__':
    x = np.linspace(-1., 1., 5).reshape(-1, 1)
    y = np.sin(x)

    kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0, variance=1.0)
    meanf = gpflow.mean_functions.Zero()
    model = gpflow.models.GPR((x, y), kernel=kernel, noise_variance=1.0)
    old_parameter = model.likelihood.variance
    new_parameter = gpflow.Parameter(
        old_parameter,
        trainable=old_parameter.trainable,
        prior=old_parameter.prior,
        name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
        transform=tfp.bijectors.Softplus(),
    )
    model.likelihood.variance = new_parameter
    print(model.log_marginal_likelihood())