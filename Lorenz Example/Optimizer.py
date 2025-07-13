# This code is used for creating models used for the paper.
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class Optimizer:
    def __init__(self, objective_function, bounds, n_iterations=100, initial_points=20, exploration_weight=0.1):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_iterations = n_iterations
        self.initial_points = initial_points
        self.exploration_weight = exploration_weight
        self.X_sample = None
        self.Y_sample = None
        self.best_params = None
        self.best_score = None

    def expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.01):
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.min(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        ei += self.exploration_weight * sigma
        return ei

    def optimize(self):
        self.X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          size=(self.initial_points, self.bounds.shape[0]))
        self.Y_sample = np.array([self.objective_function(x)[0] for x in self.X_sample])

        for i in range(self.n_iterations):
            gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25)
            gpr.fit(self.X_sample, self.Y_sample)

            res = minimize(
                lambda x: -self.expected_improvement(x.reshape(1, -1), self.X_sample, self.Y_sample, gpr),
                np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.bounds.shape[0],)),
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            X_next = res.x.reshape(1, -1)
            Y_next, additional_info = self.objective_function(X_next[0])

            self.X_sample = np.vstack((self.X_sample, X_next))
            self.Y_sample = np.append(self.Y_sample, Y_next)

            print(f"Iteration {i + 1}/{self.n_iterations}")
            print(f"Current score: {Y_next:.6f}")
            print(f"Best score so far: {np.min(self.Y_sample):.6f}")
            print(f"Parameters: {X_next[0]}")
            print("Additional info:", additional_info)
            print("---")

        best_idx = np.argmin(self.Y_sample)
        self.best_params = self.X_sample[best_idx]
        self.best_score = self.Y_sample[best_idx]

        print("Optimization completed.")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best parameters: {self.best_params}")

        return self.best_params, self.best_score
