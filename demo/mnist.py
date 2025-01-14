import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert PIL image to numpy array
        image_array = np.array(image, dtype=np.float32)
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        # Convert to tensor and add channel dimension
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        # Apply MNIST normalization
        image_tensor = (image_tensor - 0.1307) / 0.3081
        return image_tensor, label


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data():
    # Load raw datasets without transforms
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=None)
    test_dataset = datasets.MNIST("./data", train=False, transform=None)

    # Wrap datasets with our custom wrapper
    train_dataset = MNISTWrapper(train_dataset)
    test_dataset = MNISTWrapper(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    return accuracy


def main():
    # Load data
    train_loader, test_loader = load_data()

    # Initialize model and move to device
    model = MNISTConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_accuracy = 0
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "mnist_model.pth")

    print(f"Best test accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
