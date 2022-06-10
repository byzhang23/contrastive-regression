import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from jax import grad, value_and_grad
from jax import jit
from jax import vmap
import jax.random as random
import jax.scipy as scipy
from jax.example_libraries import optimizers


class LinearSSCR:
    def __init__(self):
        self.is_fitted = False

    def fit(
        self,
        X,
        Y,
        R,
        d,
        learning_rate=1e-2,
        tol=1e-4,
        max_steps=1e5,
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

        self.X = X
        self.Y = Y
        self.R = R
        self.n = n
        self.m = m
        self.p = p
        self.d = d

        self.set_up_objective()
        self.maximize_LL(
            learning_rate=learning_rate,
            tol=tol,
            max_steps=max_steps,
            verbose=verbose,
            print_every=print_every,
        )
        self.is_fitted = True

    def set_up_objective(self):

        # Minimize negative log likelihood
        self.objective = lambda params: -self.log_likelihood(
            params, self.X, self.Y, self.R
        )

    def maximize_LL(self, learning_rate, tol, max_steps, verbose, print_every):
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

        @jit
        def step(step, opt_state):
            value, grads = value_and_grad(self.objective)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        last_mll = onp.inf
        for step_num in range(int(max_steps)):
            curr_mll, opt_state = step(step_num, opt_state)
            if jnp.abs(last_mll - curr_mll) < tol:
                break
            last_mll = curr_mll
            if verbose and step_num % print_every == 0:
                print(
                    "Step: {:<15} Log likelihood: {}".format(
                        step_num, onp.round(-1 * onp.asarray(curr_mll), 2)
                    )
                )

        fitted_params = get_params(opt_state)

        self.S = fitted_params["S"]
        self.W = fitted_params["W"]
        self.beta = fitted_params["beta"]
        self.sigma_sq = jnp.exp(fitted_params["sigma_sq"])
        self.tau_sq = jnp.exp(fitted_params["tau_sq"])

        # Transformed parameters
        self.A = self.S @ self.S.T + self.sigma_sq * jnp.eye(self.d)
        self.B = (
            (self.beta @ self.beta.T) / self.tau_sq
            + (self.W @ self.W.T) / self.sigma_sq
            - (self.W @ self.S.T @ jnp.linalg.solve(self.A, self.S @ self.W.T))
            / self.sigma_sq
            + jnp.eye(self.d)
        )

        self.Ainv_S = jnp.linalg.solve(self.A, self.S)
        self.Binv_beta = jnp.linalg.solve(self.B, self.beta)

    def inner_product_vectorized(self, v, A):
        # Helps compute v^T A v quickly for many v's
        return v.T @ A @ v

    def log_likelihood(self, params, X, Y, R):

        sigma_sq = jnp.exp(params["sigma_sq"])
        tau_sq = jnp.exp(params["tau_sq"])
        # tau_sq = tau ** 2
        # sigma_sq = sigma ** 2

        # Define transformed parameters
        A = params["S"] @ params["S"].T + sigma_sq * jnp.eye(self.d)
        O = params["S"].T @ params["S"] + sigma_sq * jnp.eye(self.p)
        Q = O + params["W"].T @ params["W"]
        B = (
            (params["beta"] @ params["beta"].T) / tau_sq
            + (params["W"] @ params["W"].T) / sigma_sq
            - (
                params["W"]
                @ params["S"].T
                @ jnp.linalg.solve(A, params["S"] @ params["W"].T)
            )
            / sigma_sq
            + jnp.eye(self.d)
        )

        eta = (
            params["W"] @ self.X.T / sigma_sq
            - params["W"]
            @ params["S"].T
            @ jnp.linalg.solve(A, params["S"] @ self.X.T)
            / sigma_sq
        )

        Binv_beta = jnp.linalg.solve(B, params["beta"])
        Oinv = jnp.linalg.solve(O, jnp.eye(self.p))
        Qinv = jnp.linalg.solve(Q, jnp.eye(self.p))

        # -n/2 log(tau^2 / (tau - beta^T B^-1 beta))
        first_term = (
            -0.5
            * self.n
            * jnp.log(tau_sq ** 2 / (tau_sq - params["beta"].T @ Binv_beta))
        )

        # -(tau - beta^T B^-1 beta) / 2tau^2 * \sum_{i=1}^n (r_i - (tau beta^T B^-1 eta_i) / (tau - beta^T B^-1 beta))
        second_term_scalar = (
            -0.5 * (tau_sq - params["beta"].T @ Binv_beta) / (tau_sq ** 2)
        )
        second_term_sum = jnp.sum(
            (
                R
                - tau_sq
                * params["beta"].T
                @ jnp.linalg.solve(B, eta)
                / (tau_sq - params["beta"].T @ Binv_beta)
            )
            ** 2
        )
        second_term = second_term_scalar * second_term_sum

        # -n/2 log det(Q) - 0.5 * \sum_{i=1}^n x_i^T Q^-1 x_i
        xQx = vmap(lambda x: self.inner_product_vectorized(x, Qinv))(X)
        third_term = -0.5 * self.n * jnp.linalg.slogdet(Q)[1] - 0.5 * jnp.sum(xQx)

        # -m/2 log det(O) - 0.5 * \sum_{j=1}^m y_j^T O^-1 y_j
        yOy = vmap(lambda y: self.inner_product_vectorized(y, Oinv))(Y)
        fourth_term = -0.5 * self.m * jnp.linalg.slogdet(O)[1] - 0.5 * jnp.sum(yOy)

        # Complete log likelihood is sum of these terms
        LL = first_term + second_term + third_term + fourth_term

        # Remove singleton dimension
        return jnp.squeeze(LL)

    def predict(self, Xstar):

        if not self.is_fitted:
            raise Exception("You need to fit the model before making predictions.")

        numerator = (
            self.tau_sq
            * self.Binv_beta.T
            @ (self.W @ Xstar.T - self.W @ self.S.T @ self.Ainv_S @ Xstar.T)
        )
        denominator = self.sigma_sq * (self.tau_sq - self.beta.T @ self.Binv_beta)
        preds = numerator / denominator
        return preds.squeeze()


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

    model = LinearSSCR()
    model.fit(X, Y, R, d) #, max_steps=0)
    preds = model.predict(X)
    plt.scatter(R, preds)
    plt.show()
