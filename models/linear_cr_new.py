import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import grad, value_and_grad
from jax import jit
from jax import vmap
import jax.random as random
import jax.scipy as scipy
from jax.example_libraries import optimizers


# Class for linear contrastive regression
class LinearContrastiveRegression:

    # Constructor
    def __init__(self):
        self.is_fitted = False

    # Fit the model
    def fit(
        self,
        X,  # Foreground data matrix (n x p)
        Y,  # Background data matrix (m x p)
        R,  # Response vector (n x 1)
        d,  # Latent dimension
        seed=1, # Random seed
        learning_rate=1e-2,
        tol=1e-4,  # Optimization tolerance
        max_steps=1e6,  # Max number of optimization steps
        verbose=True,
        print_every=200,
    ):

        # X is (n x p)
        # Y is (m x p)
        # R is (n x 1)

        assert X.shape[1] == Y.shape[1]
        assert len(X) == len(R)
        n, p = X.shape
        m, _ = Y.shape
        assert d <= p

        # Store data and dimensions as class properties
        self.X = X  # Foreground matrix
        self.Y = Y  # Background matrix
        self.R = R  # Response vector
        self.n = n  # Num. FG
        self.m = m  # Num. BG
        self.p = p  # Num. of genes
        self.d = d  # Dimension of latent space

        # Set up log likelihood objective
        self.set_up_objective()

        # Maximize log likelihood
        self.maximize_LL(
            seed=seed, 
            learning_rate=learning_rate,
            tol=tol,
            max_steps=max_steps,
            verbose=verbose,
            print_every=print_every,
        )

        # Model is now fitted
        self.is_fitted = True

    # Sets up log likelihood function
    def set_up_objective(self):

        # Minimize negative log likelihood
        self.objective = lambda params: -self.log_likelihood(
            params, self.X, self.Y, self.R
        )

    # Runs optimization to maximize log likelihood
    def maximize_LL(self, seed, learning_rate, tol, max_steps, verbose, print_every):
        onp.random.seed(seed)
        params = {
            "S": 0.1 * onp.random.normal(size=(self.d, self.p)),
            "W": 0.1 * onp.random.normal(size=(self.d, self.p)),
            "beta": 0.1 * onp.random.normal(size=(self.d, 1)),
            "sigma_sq": 0.1 * onp.random.normal(size=(1)),
            "tau_sq": 0.1 * onp.random.normal(size=(1)),
        }

        # Initialize optimizer
        opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
        opt_state = opt_init(params)

        # Step function that performs one step of optimization
        # Use JIT to speed up repeated calls
        @jit
        def step(step, opt_state):
            value, grads = value_and_grad(self.objective)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        # Optimization loop
        last_mll = onp.inf
        for step_num in range(int(max_steps)):
            curr_mll, opt_state = step(step_num, opt_state)
            if jnp.abs(last_mll - curr_mll) < tol:
                break
            last_mll = curr_mll
            if verbose and step_num % print_every == 0:
                print(
                    "Step: {:<15} Log marginal lik.: {}".format(
                        step_num, onp.round(-1 * onp.asarray(curr_mll), 2)
                    )
                )

        # Store fitted parameter values
        fitted_params = get_params(opt_state)
        # true_params = {
        #     "S": onp.array([[-1,1]]),
        #     "W": onp.array([[1,1]]),
        #     "beta": onp.array([[1]]),
        #     "sigma_sq": onp.array([1e-2]),
        #     "tau_sq": onp.array([1e-2]),
        # }

        self.S = fitted_params["S"]
        self.W = fitted_params["W"]
        self.beta = fitted_params["beta"]
        self.sigma_sq = jnp.exp(fitted_params["sigma_sq"])
        self.tau_sq = jnp.exp(fitted_params["tau_sq"])

        # print('Estimated W = ',self.W)
        # print('Estimated S = ',self.S)
        # print('Estimated beta = ',self.beta)
        # print('Estimated sigma = ',self.sigma_sq)
        # print('Estimated tau = ',self.tau_sq)
        # Transformed parameters

        self.P = self.S.T @ self.S + self.sigma_sq * jnp.eye(self.p)
        # self.Pinv = jnp.linalg.solve(self.P, jnp.eye(self.p))
        self.Pinv = woodbury_inversion(
            A_diag=self.sigma_sq * jnp.ones(self.p),
            U=self.S.T,
            C_diag=jnp.ones(self.d),
            V=self.S
        )
        self.A = jnp.linalg.solve(
            self.W @ self.Pinv @ self.W.T + jnp.eye(self.d), jnp.eye(self.d)
        )

        # microergodic parameter is the one before x in the mean of the Gaussian in equation (20): beta^T A W^T P^{-1}
        # print('Estimated microergodic = ', self.beta.T @ self.A @ self.W @ self.Pinv)
        # print('True microergodic = ', true_params["beta"].T @ jnp.linalg.solve(true_params["W"] @ jnp.linalg.solve(true_params["S"].T @ true_params["S"] + true_params["sigma_sq"] * jnp.eye(self.p),jnp.eye(self.p)) @ true_params["W"].T, jnp.eye(self.d)) @ true_params["W"] @ jnp.linalg.solve(true_params["S"].T @ true_params["S"] + true_params["sigma_sq"] * jnp.eye(self.p), jnp.eye(self.p)))

        # Compute posterior mean of foreground-specific latent variables
        # self.t = (self.A @ self.W @ jnp.linalg.solve(self.P, self.X.T)).T
        self.t = (self.A @ self.W @ self.Pinv @ self.X.T).T

    def inner_product_vectorized(self, v, A):
        # Helps compute v^T A v quickly for many v's
        return v.T @ A @ v

    def log_likelihood(self, params, X, Y, R):

        sigma_sq = jnp.exp(params["sigma_sq"])
        tau_sq = jnp.exp(params["tau_sq"])
        # tau_sq = tau ** 2
        # sigma_sq = sigma ** 2

        # Define transformed parameters

        P = params["S"].T @ params["S"] + sigma_sq * jnp.eye(self.p)
        Q = P + params["W"].T @ params["W"]
        M = jnp.append(params["S"].T, params["W"].T, axis=1).T

        # Pinv = jnp.linalg.solve(P, jnp.eye(self.p))
        # Pinv = (
        #     1 / sigma_sq * jnp.eye(self.p)
        #     - 1
        #     / (sigma_sq**2)
        #     * params["S"].T
        #     @ jnp.linalg.solve(
        #         jnp.eye(self.d) + 1 / sigma_sq * params["S"] @ params["S"].T,
        #         jnp.eye(self.d),
        #     )
        #     @ params["S"]
        # )

        Pinv = woodbury_inversion(
            A_diag=sigma_sq * jnp.ones(self.p),
            U=params["S"].T,
            C_diag=jnp.ones(self.d),
            V=params["S"]
        )
        


        # Qinv = jnp.linalg.solve(Q, jnp.eye(self.p))
        Qinv = (
            1 / sigma_sq * jnp.eye(self.p)
            - 1
            / (sigma_sq**2)
            * M.T
            @ jnp.linalg.solve(
                jnp.eye(self.d * 2) + 1 / sigma_sq * M @ M.T, jnp.eye(self.d * 2)
            )
            @ M
        )
        # import ipdb; ipdb.set_trace()
        # Qinv = woodbury_inversion(
        #     A_diag=sigma_sq * jnp.ones(self.p),
        #     U=params["S"].T,
        #     C_diag=jnp.ones(self.d),
        #     V=params["S"]
        # )
        A = jnp.linalg.solve(
            params["W"] @ Pinv @ params["W"].T + jnp.eye(self.d), jnp.eye(self.d)
        )

        eta = (
            # params["beta"].T @ A @ params["W"] @ jnp.linalg.solve(P, jnp.eye(self.p)) @ self.X.T
            params["beta"].T
            @ A
            @ params["W"]
            @ Pinv
            @ self.X.T
        )

        # -n/2 log(tau^2 + beta^T A beta)
        first_term = (
            -0.5 * self.n * jnp.log(tau_sq + params["beta"].T @ A @ params["beta"])
        )

        # -1 / 2(tau^2 + beta^T A beta) * \sum_{i=1}^n (r_i - )^2
        second_term_scalar = -0.5 / (tau_sq + params["beta"].T @ A @ params["beta"])
        second_term_sum = jnp.sum((R - eta.T) ** 2)
        second_term = second_term_scalar * second_term_sum

        # -n/2 log det(Q) - 0.5 * \sum_{i=1}^n x_i^T Q^-1 x_i
        xQx = vmap(lambda x: self.inner_product_vectorized(x, Qinv))(X)
        third_term = -0.5 * self.n * jnp.linalg.slogdet(Q)[1] - 0.5 * jnp.sum(xQx)

        # -m/2 log det(P) - 0.5 * \sum_{j=1}^m y_j^T P^-1 y_j
        yPy = vmap(lambda y: self.inner_product_vectorized(y, Pinv))(Y)
        fourth_term = -0.5 * self.m * jnp.linalg.slogdet(P)[1] - 0.5 * jnp.sum(yPy)

        # Complete log likelihood is sum of these terms
        LL = first_term + second_term + third_term + fourth_term

        # Remove singleton dimension
        return jnp.squeeze(LL)

    # Make predictions for R given new foreground sample(s)
    def predict(self, Xstar):

        if not self.is_fitted:
            raise Exception("You need to fit the model before making predictions.")

        preds = self.beta.T @ self.A @ self.W @ self.Pinv @ Xstar.T

        # true_params = {
        #     "S": onp.array([[-1, 1]]),
        #     "W": onp.array([[1, 1]]),
        #     "beta": onp.array([[1]]),
        #     "sigma_sq": 1e-2,
        #     "tau_sq": 1e-2,
        # }
        # preds = true_params["beta"].T @ jnp.linalg.solve(true_params["W"] @ jnp.linalg.solve(true_params["S"].T @ true_params["S"] + jnp.eye(self.p),jnp.eye(self.p)) @ true_params["W"].T, jnp.eye(self.d)) @ true_params["W"] @ jnp.linalg.solve(true_params["S"].T @ true_params["S"] + jnp.eye(self.p), jnp.eye(self.p)) @ Xstar.T
        return preds.squeeze()


def woodbury_inversion(A_diag, U, C_diag, V):

    # (A + UCV)^{-1} = A^{-1} - A^{-1} U(C + VA^{-1}U)^{-1}VA^{-1}

    # A and C are assumed to be diagonal
    A_inv = jnp.diag(1 / A_diag)
    C_inv = jnp.diag(1 / C_diag)

    inner_mat = C_inv + V @ A_inv @ U
    inner_mat_inv = jnp.linalg.solve(inner_mat, jnp.eye(len(inner_mat)))
    overall_inv = A_inv - A_inv @ U @ inner_mat_inv @ V @ A_inv
    return overall_inv

if __name__ == "__main__":


    n = 100
    m = 100
    p = 2
    d = 1

    zx = onp.random.normal(size=(n, d))
    zy = onp.random.normal(size=(m, d))
    t = onp.random.normal(size=(n, d))
    W = onp.random.normal(size=(d, p))
    S = onp.random.normal(size=(d, p))
    beta = onp.random.normal(size=(d, 1))
    sigma = 1e-2
    tau = 1e-2

    X = zx @ S + t @ W + onp.random.normal(scale=sigma, size=(n, 2))
    Y = zy @ S + onp.random.normal(scale=sigma, size=(m, 2))
    R = t @ beta + onp.random.normal(scale=tau, size=(n, 1))

    model = LinearContrastiveRegression()
    model.fit(X, Y, R, d)
    preds = model.predict(X)
    plt.scatter(R, preds)
    plt.show()
