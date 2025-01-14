import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from src.bayesian_optimizer import (
    NeuralBayesianOptimizer,
)  # Assuming the code is in paste.py


# Create a challenging synthetic task
class ComplexNetwork(nn.Module):
    def __init__(self, hidden_size, dropout_rate, activation_slope):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LeakyReLU(activation_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(activation_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x)


def generate_dataset():
    # Create a synthetic dataset with multiple local minima
    X = torch.randn(1000, 2)
    y = torch.sin(X[:, 0] * 3) * torch.cos(X[:, 1] * 2) + 0.1 * torch.randn(1000)
    return X, y.unsqueeze(1)


def train_evaluate_model(
    model, train_X, train_y, test_X, test_y, learning_rate, batch_size, epochs=50
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        # Create mini-batches
        indices = torch.randperm(len(train_X))
        for i in range(0, len(train_X), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_X = train_X[batch_indices]
            batch_y = train_y[batch_indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_loss = criterion(test_outputs, test_y).item()
    return test_loss


def test_standard_gradient_descent():
    # Generate dataset
    train_X, train_y = generate_dataset()
    test_X, test_y = generate_dataset()

    # Fixed hyperparameters for baseline
    model = ComplexNetwork(hidden_size=64, dropout_rate=0.2, activation_slope=0.1)

    start_time = time.time()
    test_loss = train_evaluate_model(
        model, train_X, train_y, test_X, test_y, learning_rate=0.001, batch_size=32
    )
    time_taken = time.time() - start_time

    return test_loss, time_taken


def test_neural_bayesian_optimization():
    # Generate dataset
    train_X, train_y = generate_dataset()
    test_X, test_y = generate_dataset()

    # Define hyperparameter space
    hyperparameter_space = {
        "hidden_size": (32, 128, int),
        "dropout_rate": (0.0, 0.5, float),
        "activation_slope": (0.01, 0.3, float),
        "learning_rate": (0.0001, 0.01, float),
        "batch_size": (16, 64, int),
    }

    optimizer = NeuralBayesianOptimizer(hyperparameter_space)

    # Initial random points
    n_initial = 5
    initial_configs = []
    initial_losses = []

    start_time = time.time()

    # Generate and evaluate initial random configurations
    for _ in range(n_initial):
        config = optimizer.suggest_next_points(n_points=1)
        model = ComplexNetwork(
            hidden_size=config["hidden_size"],
            dropout_rate=config["dropout_rate"],
            activation_slope=config["activation_slope"],
        )
        loss = train_evaluate_model(
            model,
            train_X,
            train_y,
            test_X,
            test_y,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
        )
        initial_configs.append(config)
        initial_losses.append(loss)

    # Fit optimizer with initial data
    optimizer.fit(initial_configs, initial_losses)

    # Iterative optimization
    n_iterations = 10
    best_loss = float("inf")
    best_config = None

    for i in range(n_iterations):
        # Get next suggestion
        next_config = optimizer.suggest_next_points(n_points=1)

        # Evaluate it
        model = ComplexNetwork(
            hidden_size=next_config["hidden_size"],
            dropout_rate=next_config["dropout_rate"],
            activation_slope=next_config["activation_slope"],
        )
        loss = train_evaluate_model(
            model,
            train_X,
            train_y,
            test_X,
            test_y,
            learning_rate=next_config["learning_rate"],
            batch_size=next_config["batch_size"],
        )

        # Update optimizer
        optimizer.update_observations(next_config, loss)

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_config = next_config

    time_taken = time.time() - start_time
    return best_loss, time_taken, best_config


def main():
    # Run multiple trials
    n_trials = 5
    gd_losses = []
    gd_times = []
    nbo_losses = []
    nbo_times = []

    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")

        print("Testing standard gradient descent...")
        gd_loss, gd_time = test_standard_gradient_descent()
        gd_losses.append(gd_loss)
        gd_times.append(gd_time)

        print("Testing neural Bayesian optimization...")
        nbo_loss, nbo_time, best_config = test_neural_bayesian_optimization()
        nbo_losses.append(nbo_loss)
        nbo_times.append(nbo_time)

        print(f"\nTrial {i+1} Results:")
        print(f"GD - Loss: {gd_loss:.4f}, Time: {gd_time:.2f}s")
        print(f"NBO - Loss: {nbo_loss:.4f}, Time: {nbo_time:.2f}s")

    # Print summary statistics
    print("\nFinal Results:")
    print("\nGradient Descent:")
    print(f"Mean Loss: {np.mean(gd_losses):.4f} ± {np.std(gd_losses):.4f}")
    print(f"Mean Time: {np.mean(gd_times):.2f}s ± {np.std(gd_times):.2f}s")

    print("\nNeural Bayesian Optimization:")
    print(f"Mean Loss: {np.mean(nbo_losses):.4f} ± {np.std(nbo_losses):.4f}")
    print(f"Mean Time: {np.mean(nbo_times):.2f}s ± {np.std(nbo_times):.2f}s")


if __name__ == "__main__":
    main()
