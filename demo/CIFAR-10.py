import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
import PIL.Image
from src.latin_hypercube_sampling import latin_hypercube_sampling
from src.bayesian_optimizer import NeuralBayesianOptimizer
import json


class CIFAR10Trainer:
    def __init__(self, data_dir="./data", device=None):
        self.data_dir = Path(data_dir)
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._setup_data_transforms()
        self._load_datasets()

    def _setup_data_transforms(self):
        """Setup basic data transforms for CIFAR-10 with explicit PIL Image conversion."""
        self.train_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: PIL.Image.fromarray(x) if isinstance(x, np.ndarray) else x
                ),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: PIL.Image.fromarray(x) if isinstance(x, np.ndarray) else x
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def _load_datasets(self):
        """Load and split datasets."""
        try:
            self.train_dataset = datasets.CIFAR10(
                self.data_dir, train=True, download=True, transform=self.train_transform
            )
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, download=True, transform=self.test_transform
            )
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

    def _create_data_loaders(self, batch_size, val_split=0.1):
        """Create train/val/test data loaders with given batch size."""
        train_size = len(self.train_dataset)
        indices = list(range(train_size))
        split = int(np.floor(val_split * train_size))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        try:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True,
            )
            val_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            raise

        return train_loader, val_loader, test_loader

    def _create_model(self, dropout_rate):
        """Create a simple CNN model."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10),
        )
        return model.to(self.device)

    def train_and_evaluate(self, hyperparameters, epochs=10):
        """Train model with given hyperparameters and return validation accuracy."""
        # Extract hyperparameters
        lr = hyperparameters["learning_rate"]
        batch_size = hyperparameters["batch_size"]
        dropout_rate = hyperparameters["dropout_rate"]
        weight_decay = hyperparameters["weight_decay"]

        # Create data loaders
        try:
            train_loader, val_loader, _ = self._create_data_loaders(batch_size)
        except Exception as e:
            print(f"Failed to create data loaders: {e}")
            return 0.0

        # Create model and optimizer
        model = self._create_model(dropout_rate)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training phase
            model.train()
            i = 0
            for inputs, targets in train_loader:
                print(inputs.shape[0])
                breakpoint()
                try:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"Error during training step: {e}")
                    continue
            print(i)
            exit()

            # Validation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    try:
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        print(f"Error during validation step: {e}")
                        continue

            if total > 0:  # Avoid division by zero
                val_acc = 100.0 * correct / total
                best_val_acc = max(best_val_acc, val_acc)
                print(f"Epoch {epoch+1}/{epochs}: Validation Accuracy = {val_acc:.2f}%")

        return best_val_acc


def test_objective():
    """Test the trainer with some sample hyperparameters."""
    trainer = CIFAR10Trainer()

    sample_hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
    }

    try:
        accuracy = trainer.train_and_evaluate(sample_hyperparameters, epochs=2)
        print(f"Final validation accuracy: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error in test_objective: {e}")


def evaluate_hp(hpcs):
    """Test the trainer with some sample hyperparameters."""
    trainer = CIFAR10Trainer()

    configs = []
    results = []
    for hpc in hpcs:
        accuracy = trainer.train_and_evaluate(hpc, epochs=5)
        configs.append(hpc)
        results.append(accuracy)

        print(f"Final validation accuracy: {accuracy:.2f}%")

    return hpcs, results


if __name__ == "__main__":
    bounds = [
        [0.00001, 0.1],
        [32, 1024],
        [0.0, 0.9],
        [1e-6, 1e-2],
    ]

    hyperparameter_space = {
        "learning_rate": (0.00001, 0.1, float),
        "batch_size": (32, 1024, int),
        "dropout_rate": (0.0, 0.9, float),
        "weight_decay": (1e-6, 1e-2, float),
    }

    results = []
    hpcs = latin_hypercube_sampling(hyperparameter_space, 2)
    X_d, results_d = evaluate_hp(hpcs)
    Xs = torch.tensor([list(X.values()) for X in X_d])
    Ys = torch.tensor(results_d)
    optimizer = NeuralBayesianOptimizer(hyperparameter_space)
    print(Xs.shape, Ys.shape)
    optimizer.fit(Xs, Ys)

    final_results = []
    for i in range(15):
        new_points = optimizer.suggest_next_points(n_points=2)
        Xp, resultsp = evaluate_hp(new_points)
        optimizer.update_observations(Xp, resultsp)
        print(Xp, resultsp)
        final_results.append((Xp, resultsp))
    with open("./results.json", "w") as f:
        json.dump(final_results, f)
