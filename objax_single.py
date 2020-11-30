import jax
import jax.numpy as np
import jax.random as jr
import objax
import numpy as onp
import matplotlib.pyplot as plt
from gpblocks import Prior, RBF, Gaussian
onp.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__':
    # Settings
    lr = 0.01  # learning rate
    batch = 256
    epochs = 50
    N = 50
    sigma = 0.1

    # Data simulate
    onp.random.seed(0)
    X = np.linspace(-1., 1., N)
    y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(
        4.0 * X)
    y += sigma * onp.random.randn(N)
    y -= np.mean(y)
    y /= np.std(y)
    Xtest = np.linspace(-1.5, 1.5, 500)

    gp_model = Prior(kernel=RBF(), jitter=1e-5)
    dist = gp_model.forward(X)
    print(gp_model.vars())

    #  Test __mul__
    gp_model = gp_model * Gaussian()

    key = jr.PRNGKey(0)

    # samples = dist.sample(10, key)
    # plt.plot(samples.T)
    # plt.show()


    def loss(X, label):
        distr = gp_model.forward(X)
        return -distr.log_prob(label).mean()

    opt = objax.optimizer.SGD(gp_model.vars())
    gv = objax.GradValues(loss, gp_model.vars())

    def train_op(x, label):
        g, v = gv(x, label)
        opt(lr, g)
        return v

    # This line is optional: it is compiling the code to make it faster.
    train_op = objax.Jit(train_op, gv.vars() + opt.vars())

    losses = []
    for epoch in range(epochs):
        # Train
        loss = train_op(X, y.squeeze())
        losses.append(loss)

    # mu, cov = posterior(X, X, y.squeeze(), jax.nn.softplus(gp_model.likelihood.noise.value), gp_model)
    mu, cov = gp_model.predict(Xtest, X, y.squeeze())
    plt.plot(X, y, 'o')
    plt.plot(Xtest, mu)
    plt.fill_between(
        Xtest.ravel(),
        mu.squeeze() - 1.96 * np.sqrt(
            np.diag(cov) + jax.nn.softplus(gp_model.likelihood.noise.value)),
        mu.squeeze() + 1.96 * np.sqrt(
            np.diag(cov) + jax.nn.softplus(gp_model.likelihood.noise.value)),
        alpha=0.5)
    plt.show()
