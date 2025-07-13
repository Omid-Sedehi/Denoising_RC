import numpy as np
from scipy import linalg
import networkx as nx
import copy
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from Optimizer import Optimizer

class ReservoirComputer:
    def __init__(self, n_inputs, n_outputs, n_nodes=200, spectral_radius=0.95,
                 leaking_rate=0.3, input_scaling=1.0, bias_scaling=1.0,
                 connectivity=0.1, random_state=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.connectivity = connectivity
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self._initialize_weights()
        self.W_out = None

    def _initialize_weights(self):
        self.W_in = np.random.rand(self.n_nodes, self.n_inputs + 1) - 0.5
        self.W_in *= self.input_scaling

        self.W = np.random.rand(self.n_nodes, self.n_nodes) - 0.5
        mask = np.random.rand(*self.W.shape) < self.connectivity
        self.W *= mask

        radius = np.max(np.abs(linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / radius

    def _update_state(self, state, input_data):
        preactivation = np.dot(self.W, state) + np.dot(self.W_in, np.hstack([input_data, 1]))
        return (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(preactivation)

    def fit(self, X, y, washout=100, reg_param=1e-8):
        n_samples, _ = X.shape
        extended_states = np.zeros((n_samples - washout, self.n_nodes + 1))

        state = np.zeros(self.n_nodes)
        for t in range(n_samples):
            state = self._update_state(state, X[t])
            if t >= washout:
                extended_states[t - washout] = np.hstack([state, 1])

        A = extended_states
        B = y[washout:]
        model_rigde = RidgeCV(alphas=[1e-16, 1e-15, 1e-14, 1e-13,  1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                                      1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8], fit_intercept=False)
        model_rigde.fit(A, B)
        self.W_out = model_rigde.coef_.T
        # self.W_out = np.dot(np.linalg.inv(np.dot(A.T, A) + reg_param * np.eye(A.shape[1])), np.dot(A.T, B))

    def predict(self, X):
        n_samples, _ = X.shape
        predictions = np.zeros((n_samples, self.n_outputs))
        state = np.zeros(self.n_nodes)

        for t in range(n_samples):
            state = self._update_state(state, X[t])
            predictions[t] = np.dot(np.hstack([state, 1]), self.W_out)

        return predictions

    def prune(self, params):
        prune_percent, activation_weight, variance_weight, connectivity_weight, clustering_weight, pagerank_weight = params

        states = self._get_states(self.W_in[:, :-1])
        avg_activation = np.mean(np.abs(states), axis=0)
        variance = np.var(states, axis=0)

        G = nx.from_numpy_array(np.abs(self.W))
        connectivity = np.array([d for _, d in G.degree()])
        clustering = np.array(list(nx.clustering(G).values()))
        pagerank = np.array(list(nx.pagerank(G).values()))

        properties = [avg_activation, variance, connectivity, clustering, pagerank]
        normalized_properties = [(p - np.min(p)) / (np.max(p) - np.min(p)) for p in properties]

        weights = [activation_weight, variance_weight, connectivity_weight, clustering_weight, pagerank_weight]
        importance = sum(w * p for w, p in zip(weights, normalized_properties))

        n_keep = int(self.n_nodes * (1 - prune_percent))
        keep_indices = np.argsort(importance)[-n_keep:]

        self.W = self.W[keep_indices][:, keep_indices]
        self.W_in = self.W_in[keep_indices]
        if self.W_out is not None:
            self.W_out = self.W_out[:-1][keep_indices]

        self.n_nodes = n_keep

    def _get_states(self, inputs):
        states = np.zeros((inputs.shape[0], self.n_nodes))
        state = np.zeros(self.n_nodes)
        for t, input_data in enumerate(inputs):
            state = self._update_state(state, input_data)
            states[t] = state
        return states

    def one_shot_pruning(self, X_train, y_train, X_val, y_val, n_iterations=100, initial_points=20):
        def pruning_objective(params):
            pruned_rc = copy.deepcopy(self)
            pruned_rc.prune(params)
            pruned_rc.fit(X_train, y_train)
            y_pred = pruned_rc.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            complexity_penalty = params[0] * (self.n_nodes / pruned_rc.n_nodes)
            score = mse + 0.05 * complexity_penalty
            return score, {"pruned_nodes": pruned_rc.n_nodes}

        pruning_bounds = [
            (0.1, 0.9),  # prune_percent
            (0, 1),  # activation_weight
            (0, 1),  # variance_weight
            (0, 1),  # connectivity_weight
            (0, 1),  # clustering_weight
            (0, 1)  # pagerank_weight
        ]

        optimizer = Optimizer(pruning_objective, pruning_bounds, n_iterations=n_iterations,
                                      initial_points=initial_points)
        best_params, best_score = optimizer.optimize()

        # Apply the best pruning strategy
        self.prune(best_params)
        self.fit(X_train, y_train)  # Retrain the pruned reservoir

        return best_params, best_score

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_iterations=100, initial_points=20):
        def hyperparameter_objective(params):
            n_nodes, spectral_radius, leaking_rate, input_scaling, connectivity = params

            # Create a new reservoir with the given hyperparameters
            tuned_rc = ReservoirComputer(
                n_inputs=self.n_inputs,
                n_outputs=self.n_outputs,
                n_nodes=int(n_nodes),
                spectral_radius=spectral_radius,
                leaking_rate=leaking_rate,
                input_scaling=input_scaling,
                connectivity=connectivity,
                random_state=self.random_state
            )

            # Train the reservoir
            tuned_rc.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = tuned_rc.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)

            return mse, {"n_nodes": int(n_nodes)}

        # Define bounds for hyperparameters
        hyperparameter_bounds = [
            (500, 500),  # n_nodes
            (0.01, 1),  # spectral_radius
            (0.01, 1.0),  # leaking_rate
            (0.1, 2.0),  # input_scaling
            (0.1, 0.9)  # connectivity
        ]

        optimizer = Optimizer(hyperparameter_objective, hyperparameter_bounds,
                                      n_iterations=n_iterations, initial_points=initial_points)
        best_params, best_score = optimizer.optimize()

        # Update the reservoir with the best hyperparameters
        self.n_nodes = int(best_params[0])
        self.spectral_radius = best_params[1]
        self.leaking_rate = best_params[2]
        self.input_scaling = best_params[3]
        self.connectivity = best_params[4]

        # Reinitialize the reservoir with new hyperparameters
        self._initialize_weights()

        # Retrain the reservoir
        self.fit(X_train, y_train)

        return best_params, best_score

    def get_nodes(self):
        return self.n_nodes
