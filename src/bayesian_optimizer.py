import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from scipy.optimize import minimize


class HyperparameterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], feature_dim=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = dim

        # Final feature layer (no activation - we want continuous features)
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BayesianLinearRegression:
    def __init__(self, input_dim, prior_mean=None, prior_cov=None, noise_var=1.0):
        self.input_dim = input_dim
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(input_dim)
        self.prior_cov = prior_cov if prior_cov is not None else np.eye(input_dim)
        self.noise_var = noise_var

        self.posterior_mean = self.prior_mean.copy()
        self.posterior_cov = self.prior_cov.copy()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Posterior precision matrix
        precision = np.linalg.inv(self.prior_cov) + X.T @ X / self.noise_var

        # Posterior covariance
        self.posterior_cov = np.linalg.inv(precision)

        # Posterior mean
        self.posterior_mean = self.posterior_cov @ (
            np.linalg.inv(self.prior_cov) @ self.prior_mean + X.T @ y / self.noise_var
        )

    def predict(self, X):
        X = np.asarray(X)
        pred_mean = X @ self.posterior_mean
        pred_var = np.diag(
            X @ self.posterior_cov @ X.T + self.noise_var * np.eye(len(X))
        )
        return pred_mean, pred_var


def expected_improvement(X, model, best_f, xi=0.01):
    mu, sigma = model.predict(X)
    sigma = np.sqrt(sigma)

    with np.errstate(divide="warn"):
        imp = mu - best_f - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


class NeuralBayesianOptimizer:
    def __init__(
        self,
        hyperparameter_space,
        feature_dim=10,
        hidden_dims=[64, 32],
        prior_mean=None,
        prior_cov=None,
        noise_var=1.0,
    ):
        self.hp_space = hyperparameter_space
        self.hp_names = list(hyperparameter_space.keys())
        self.hp_dim = len(self.hp_names)
        self.feature_dim = feature_dim
        self.best_observed_value = -float("inf")

        # Store history of observations
        self.history_X = []
        self.history_y = []

        # Create bounds array for internal use
        self.bounds = np.array(
            [[hp_space[0], hp_space[1]] for hp_space in hyperparameter_space.values()]
        )
        self.hp_types = [hp_space[2] for hp_space in hyperparameter_space.values()]

        # Store log-scale flags for each parameter
        self.log_scale = np.array(
            [
                hp_space[0] > 0 and hp_space[1] / hp_space[0] > 100
                for hp_space in hyperparameter_space.values()
            ]
        )

        # Transform bounds to log scale where appropriate
        self.internal_bounds = self.bounds.copy()
        self.internal_bounds[self.log_scale] = np.log10(self.bounds[self.log_scale])

        # Initialize neural network
        self.feature_network = HyperparameterNetwork(
            input_dim=self.hp_dim, hidden_dims=hidden_dims, feature_dim=feature_dim
        )

        # Initialize Bayesian linear regression
        self.blr = BayesianLinearRegression(
            input_dim=feature_dim,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            noise_var=noise_var,
        )

    def _internal_to_external_scale(self, x):
        """Convert internal representation to actual hyperparameter values"""
        x_scaled = x.copy()
        # Convert from log scale where appropriate
        x_scaled[:, self.log_scale] = 10 ** x_scaled[:, self.log_scale]

        # Apply types and create dictionary
        hp_dicts = []
        for row in x_scaled:
            hp_dict = {}
            for i, (name, value, hp_type) in enumerate(
                zip(self.hp_names, row, self.hp_types)
            ):
                if hp_type == int:
                    value = int(round(value))
                hp_dict[name] = value
            hp_dicts.append(hp_dict)

        return hp_dicts if len(hp_dicts) > 1 else hp_dicts[0]

    def _external_to_internal_scale(self, hp_dicts):
        """Convert hyperparameter dictionaries to internal representation"""
        if not isinstance(hp_dicts, list):
            hp_dicts = [hp_dicts]

        x = np.zeros((len(hp_dicts), self.hp_dim))
        for i, hp_dict in enumerate(hp_dicts):
            for j, name in enumerate(self.hp_names):
                x[i, j] = hp_dict[name]

        # Convert to log scale where appropriate
        x[:, self.log_scale] = np.log10(x[:, self.log_scale])
        return x

    def train_feature_network(self, X, y, epochs=100, batch_size=32, lr=1e-3):
        """Train the neural network to predict performance from hyperparameters"""
        # Convert to internal scale if needed
        if isinstance(X, dict) or (
            isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict)
        ):
            X = self._external_to_internal_scale(X)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.feature_network.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.feature_network.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Get features and predict
                features = self.feature_network(batch_X)

                # Simple linear layer for final prediction
                pred = features.mean(dim=1)

                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def extract_features(self, X):
        """Extract features from hyperparameters using the neural network"""
        if isinstance(X, dict) or (isinstance(X, list) and isinstance(X[0], dict)):
            X = self._external_to_internal_scale(X)

        self.feature_network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            features = self.feature_network(X_tensor)
        return features.numpy()

    def fit(self, X, y):
        """Train both the feature network and Bayesian linear regression"""
        # Convert y to numpy if it's a tensor
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)

        # Reset history
        self.history_X = []
        self.history_y = []

        # Handle different input types
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
            # Convert numpy array to list of dicts
            self.history_X = [
                {name: float(x[i]) for i, name in enumerate(self.hp_names)} for x in X
            ]
            self.history_y = list(y)
        elif isinstance(X, dict):
            self.history_X = [X]
            self.history_y = [y if np.isscalar(y) else y[0]]
        elif isinstance(X, list):
            self.history_X = X
            self.history_y = list(y)

        # Ensure lengths match
        assert len(self.history_X) == len(
            self.history_y
        ), f"Mismatch in history lengths: X={len(self.history_X)}, y={len(self.history_y)}"

        # Train on full history
        self.train_feature_network(self.history_X, np.array(self.history_y))

        # Extract features for all points
        features = self.extract_features(self.history_X)

        # Fit Bayesian linear regression on all data
        self.blr.fit(features, self.history_y)

        # Update best observed value
        self.best_observed_value = max(self.best_observed_value, float(np.max(y)))

    def predict(self, X):
        """Get predictions with uncertainty"""
        features = self.extract_features(X)
        return self.blr.predict(features)

    def suggest_next_points(self, n_points=1, n_restarts=5):
        """Suggest next hyperparameter configurations to evaluate"""
        best_points = np.zeros((n_points, self.hp_dim))

        def feature_objective(x):
            """Wrapper to work with feature extraction"""
            features = self.extract_features(x.reshape(1, -1))
            return -expected_improvement(features, self.blr, self.best_observed_value)

        for i in range(n_points):
            best_x = None
            best_ei = -1

            # Multiple random starts
            for _ in range(n_restarts):
                x0 = np.random.uniform(
                    self.internal_bounds[:, 0],
                    self.internal_bounds[:, 1],
                    size=self.hp_dim,
                )
                result = minimize(
                    feature_objective,
                    x0,
                    bounds=self.internal_bounds,
                    method="L-BFGS-B",
                )

                if result.fun < best_ei:
                    best_ei = result.fun
                    best_x = result.x

            if best_x is not None:
                best_points[i] = best_x
            else:
                # If optimization failed, sample randomly
                best_points[i] = np.random.uniform(
                    self.internal_bounds[:, 0],
                    self.internal_bounds[:, 1],
                    size=self.hp_dim,
                )

        # Convert to labeled dictionary format
        return self._internal_to_external_scale(best_points)

    def update_observations(self, X_new, y_new, epochs=50):
        """Update the model with new observations"""
        # Convert y to numpy if it's a tensor
        if torch.is_tensor(y_new):
            y_new = y_new.detach().cpu().numpy()
        elif isinstance(y_new, list):
            y_new = np.array(y_new)

        # Add to history - ensure X and y match in length
        if isinstance(X_new, dict):
            self.history_X.append(X_new)
            self.history_y.append(y_new if np.isscalar(y_new) else y_new[0])
        elif isinstance(X_new, list):
            self.history_X.extend(X_new)
            self.history_y.extend(list(y_new))

        # Ensure lengths match
        assert len(self.history_X) == len(
            self.history_y
        ), f"Mismatch in history lengths: X={len(self.history_X)}, y={len(self.history_y)}"

        # Train on full history
        self.train_feature_network(
            self.history_X, np.array(self.history_y), epochs=epochs
        )

        # Extract updated features for all points
        features = self.extract_features(self.history_X)

        # Update Bayesian linear regression with all data
        self.blr.fit(features, self.history_y)

        # Update best observed value
        self.best_observed_value = max(self.best_observed_value, float(np.max(y_new)))


if __name__ == "__main__":
    # Example usage
    hyperparameter_space = {
        "learning_rate": (0.00001, 0.1, float),
        "batch_size": (32, 1024, int),
        "dropout_rate": (0.0, 0.9, float),
        "weight_decay": (1e-6, 1e-2, float),
    }

    # Generate some synthetic data
    np.random.seed(42)
    n_samples = 10

    X_initial = [
        {
            "learning_rate": np.random.uniform(1e-5, 1e-1),
            "batch_size": int(np.random.uniform(32, 1024)),
            "dropout_rate": np.random.uniform(0, 0.9),
            "weight_decay": np.random.uniform(1e-6, 1e-2),
        }
        for _ in range(n_samples)
    ]

    # Synthetic performance metric
    y_initial = np.random.uniform(0.5, 0.95, size=n_samples)

    # Initialize and train optimizer
    optimizer = NeuralBayesianOptimizer(hyperparameter_space)
    optimizer.fit(X_initial, y_initial)

    # Get suggestions
    next_points = optimizer.suggest_next_points(n_points=3)
    print("\nSuggested hyperparameters:")
    for point in next_points:
        print(point)
