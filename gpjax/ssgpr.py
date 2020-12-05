import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


class RFF:
    # initialize lengthscales and amplitude to 1
    def __init__(self, dimensions, features):
        self.D = dimensions
        self.M = features
        self.l = np.ones(self.D)
        self.sigma = 1.0
        self.w = np.random.normal(size=(self.M, self.D))
        self.scale_frequencies()
        self.grad_matrix = None
        return

    # allow user to update lengthscales to desired values
    def update_lengthscales(self, lengthscales):
        assert len(
            lengthscales
        ) == self.D, 'Lengthscale vector does not agree with dimensionality'
        self.l = lengthscales
        self.scale_frequencies()
        return

    # allow user to update the amplitude to desired value
    def update_amplitude(self, amplitude):
        self.sigma = amplitude
        return

    # allow user to update the frequencies, note that the latter has to be in a vector format
    # (for compatibility with the SSGPR algorithm)
    def update_frequencies(self, w):
        self.w = w.reshape((self.M, self.D))
        self.scale_frequencies()
        self.grad_matrix = None
        return

    # Function to scale the frequencies with the lengthscale
    def scale_frequencies(self):
        self.W = (1.0 / self.l) * self.w
        self.grad_matrix = None
        return

    # build design matrix phi
    def design_matrix(self, x):
        N = x.shape[0]
        phi_x = np.zeros((N, 2 * self.M))
        phi_x[:, :self.M] = np.cos(x.dot(self.W.T))
        phi_x[:, self.M:] = np.sin(x.dot(self.W.T))
        print("Omega: ", self.W.shape)
        print("Phi:", phi_x.shape)
        return phi_x

    # gradient of design matrix with respect to a lengthscale parameter
    def gradients_l(self, parameter, x):
        N = x.shape[0]

        if self.grad_matrix is None:
            self.grad_matrix = np.zeros((N, 2 * self.M))
            self.grad_matrix[:, :self.M] = -np.sin(x.dot(self.W.T))
            self.grad_matrix[:, self.M:] = np.cos(x.dot(self.W.T))

        dl_dli = np.zeros(self.D)
        dl_dli[parameter] = -self.l[parameter]**-2
        grad = np.zeros((N, 2 * self.M))
        grad[:, :self.M] = self.grad_matrix[:, :self.M] * (
            (x * dl_dli).dot(self.w.T))
        grad[:, self.M:] = self.grad_matrix[:, self.M:] * (
            (x * dl_dli).dot(self.w.T))
        return grad

    # gradients of design matrix with respect to freqeuncies (returned as sparse column matrix)
    def gradients_w(self, parameter, x):
        N = x.shape[0]
        i = parameter / self.D
        j = parameter - (i + 1) * self.D

        if self.grad_matrix is None:
            self.grad_matrix = np.zeros((N, 2 * self.M))
            self.grad_matrix[:, :self.M] = -np.sin(x.dot(self.W.T))
            self.grad_matrix[:, self.M:] = np.cos(x.dot(self.W.T))

        dw_dwij = x[:, int(j)] * (1.0 / self.l[int(j)])
        data = np.zeros(2 * N)
        data[:N] = self.grad_matrix[:, parameter] * dw_dwij
        data[N:] = self.grad_matrix[:, parameter + self.M] * dw_dwij
        row = np.array(list(range(N)) + list(range(N)))
        col = np.array(([parameter] * N) + [parameter + self.M] * N)
        grad = csc_matrix((data, (row, col)), shape=(N, 2 * self.M))
        return grad


class SSGPR:
    def __init__(self, x_train, y_train, basis_functions):
        self.basis_functions = basis_functions  # number of basis functions to use
        self.x_train = x_train  # store training data set
        self.y_train = y_train  # store training targets
        self.rff = RFF(x_train.shape[1],
                       self.basis_functions)  # initialize the RFF
        self.noise = 1.0  # spherical noise parameter
        self.N, self.D = x_train.shape
        self.params = None
        self.R = None
        self.Ry = None
        self.inv = None
        return

    # function to make predictions on training points x (x must be in array format)
    def predict(self, x):
        phi = self.rff.design_matrix(
            self.x_train)  # Compute training design matrix
        norm = ((self.rff.sigma**2) / self.basis_functions
                )  # comonly used constant
        R = np.linalg.cholesky(norm * np.dot(phi, phi.T) \
                               + (self.noise ** 2) * np.eye(
            self.N))  # peform cholesky factorisation of approximate Gram matrix
        Ry = np.linalg.solve(
            R, self.y_train)  # Avoid inversions by solving linear system
        print("Ry: ", Ry.shape)
        alpha = np.linalg.solve(
            R.T, Ry)  # Avoid inversions by solving linear system
        phi_s = self.rff.design_matrix(
            x)  # Compute design matrix between training and testing points
        mu = norm * phi_s.dot(phi.T).dot(alpha)  # Compute the conditional mean

        beta = np.linalg.solve(R, np.dot(
            phi, phi_s.T))  # Avoid inversions by solving linear system
        var = np.sqrt(norm * np.diag(np.dot(phi_s, phi_s.T) - norm \
                                     * np.dot(beta.T, beta))).reshape(-1,
                                                                      1)  # Compute the conditional point by point standard deviations

        return (mu, var)  # Return the mean and std as a tuple

    # function that computes the marginal likelihood
    def marginal_likelihood(self):
        phi = self.rff.design_matrix(self.x_train)
        print(phi.shape)
        if self.R is None:
            self.R = np.linalg.cholesky(((self.rff.sigma ** 2) / self.basis_functions) \
                                        * np.dot(phi, phi.T) + (self.noise ** 2) * np.eye(self.N))
            self.Ry = np.linalg.solve(self.R, self.y_train)
        if self.Ry is None:
            self.Ry = np.linalg.solve(self.R, self.y_train)

        print("Ry: ", self.R.shape)
        return 0.5 * np.sum(self.Ry**2) + np.sum(np.log(np.diag(
            self.R))) + 0.5 * self.N * np.log(2 * np.pi)

    # function that computes gradients
    def gradients(self, params):
        grad = np.zeros(len(params))  # initialize the gradient vector

        # check whther parameters are updated
        if not np.array_equal(params, self.params):
            self.update_parameters(params)

        phi = self.rff.design_matrix(self.x_train)  # get design matrix
        norm = ((self.rff.sigma**2) / self.basis_functions)  # common constant

        # if the cholesky decomposition has not been computed yet then we must compute the following
        if self.R is None:
            self.R = np.linalg.cholesky(norm * np.dot(phi, phi.T) \
                                        + (self.noise ** 2) * np.eye(self.N))
            self.inv = np.linalg.solve(self.R.T,
                                       np.linalg.solve(self.R, np.eye(self.N)))
            self.alpha = np.dot(self.inv, self.y_train)
            self.inv = np.dot(self.alpha, self.alpha.T) - self.inv

        # if cholesky was already computed but the inv matrix wasn't, then we must compute the following
        if self.inv is None:
            self.inv = np.linalg.solve(self.R.T,
                                       np.linalg.solve(self.R, np.eye(self.N)))
            self.alpha = np.dot(self.inv, self.y_train)
            self.inv = np.dot(self.alpha, self.alpha.T) - self.inv

            # gradient of the RFF amplitude
        grad[0] = -0.5 * np.trace(
            np.dot(self.inv, (2 * norm / self.rff.sigma) * np.dot(phi, phi.T)))

        # gradient of the lengthscales
        for i in range(self.D):
            a = np.dot(self.rff.gradients_l(i, self.x_train), phi.T)
            g = norm * (a + a.T)
            grad[i + 1] = -0.5 * np.trace(np.dot(self.inv, g))

        # gradient of the noise
        grad[self.D + 1] = -0.5 * np.trace(
            np.dot(self.inv, 2 * self.noise * np.eye(self.N)))

        # gradient of the frequencies
        for i in range(self.D * self.basis_functions):
            a = self.rff.gradients_w(i, self.x_train).dot(phi.T)
            g = norm * (a + a.T)
            grad[self.D + 2 + i] = -0.5 * np.trace(np.dot(self.inv, g))
        return grad

    def update_parameters(self, params):
        self.rff.update_amplitude(params[0])  # update RFF amplitude
        self.noise = params[self.D + 1]  # update the SSGPR noise variance
        self.rff.update_frequencies(
            params[self.D + 2:])  # update the RFF spectral frequencies
        self.rff.update_lengthscales(params[1:self.D +
                                            1])  # update RFF lengthscales
        self.params = params
        self.R = None
        self.Ry = None
        self.inv = None
        return

    def objective_function(self, params):
        if not np.array_equal(params, self.params):
            self.update_parameters(params)
        return self.marginal_likelihood()

    def optimize(self, restarts=1, method='CG', iterations=1000, verbose=True):
        if verbose:
            print('***************************************************')
            print('*              Optimizing parameters              *')
            print('***************************************************')

        global_opt = np.inf  # Initialize the global optimum value

        for res in range(restarts):
            # try different initializations for spectral frequencies
            opt_ml = np.inf
            for i in range(100):
                w = np.random.normal(size=self.D * self.basis_functions)
                self.rff.update_frequencies(w)
                self.rff.scale_frequencies()
                ml = self.marginal_likelihood()
                if ml < opt_ml:
                    w_opt = w

            # create parameter vector and set spectral frequencies to the best ones found
            params = np.ones(self.D * self.basis_functions + self.D + 2)
            params[0] = self.rff.sigma
            params[1:self.D + 1] = self.rff.l
            params[self.D + 1] = self.noise
            params[self.D + 2:] = w_opt.flatten()

            # optimize the objective
            opt = minimize(self.objective_function,
                           params,
                           jac=self.gradients,
                           method=method,
                           options={'maxiter': iterations})

            # check if the local optimum beat the current global optimum
            if opt['fun'] < global_opt:
                global_opt = opt['fun']  # update the global optimum
                self.opt = opt  # save a copy of the scipy optimization object

            # print out optimization result if the user wants
            if verbose:
                print(('restart # %i, negative log-likelihood = %.6f' %
                       (res + 1, opt['fun'])))

            # randomize paramters for next iteration
            if res < restarts - 1:
                self.rff.update_amplitude(np.random.normal())
                self.rff.update_lengthscales(np.random.normal(size=self.D))
                self.noise = np.random.normal()
        # update all the paramters to the optimum
        self.update_parameters(self.opt['x'])

    # plot training data along with GP
    def plot(self):
        # ensure data is only 1D
        assert self.D == 1, 'Plot can only work for 1D data'

        # set up the range of the plot
        x_max = np.max(self.x_train)
        x_min = np.min(self.x_train)
        padding = (x_max - x_min) * 0.15

        x = np.linspace(x_min - padding, x_max + padding, 1000).reshape(-1, 1)

        # get GP mean and standard deviations over the range
        mu, std = self.predict(x)

        # plot training points along with the GP
        plt.figure()
        plt.clf()
        plt.plot(x, mu, 'g', label='GP Mean')
        plt.fill_between(x.flatten(), (mu + 2 * std).flatten(),
                         (mu - 2 * std).flatten(),
                         interpolate=True,
                         alpha=0.25,
                         color='orange',
                         label='95% Confidence')
        plt.scatter(self.x_train, self.y_train, label='Training Points')
        plt.xlabel('X', fontsize=20)
        plt.ylabel('Y', fontsize=20)
        plt.legend()
        plt.show()
        return


if __name__ == '__main__':

    def sinc(N, noise=0.01):
        x = np.sort(np.random.uniform(-1, 6, size=N)).reshape(-1, 1)
        y = np.sinc(x) + np.random.normal(scale=noise, size=N).reshape(-1, 1)
        return x, y

    x_train, y_train = sinc(25, noise=0.1)

    sinc_model = SSGPR(x_train, y_train, 10)
    # sinc_model.plot()

    sinc_model.optimize(restarts=1, iterations=1, method='CG', verbose=True)
    # sinc_model.plot()
