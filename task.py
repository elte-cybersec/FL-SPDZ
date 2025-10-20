import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def load_data(client_number, label_flip=False):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Select client-specific subset
    indices = list(range(client_number * 4000, (client_number + 1) * 4000))
    dataset = Subset(mnist_train, indices)

    # Apply label flipping if this client is malicious
    if label_flip:
        print(f"[Client {client_number}] Label flipping applied.")
        # Overwrite dataset.targets (access through the underlying dataset)
        for idx in indices:
            original_label = mnist_train.targets[idx].item()
            flipped_label = (original_label + 5) % 10
            mnist_train.targets[idx] = flipped_label

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    return train_loader, test_loader



# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_model():
    return SimpleNN()

# Training function
def train(model, loader, epochs=5, lr=0.01, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        model.train()  # Set the model to training mode
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    return model


# Testing function
def test(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct, total, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")
    return total_loss / len(test_loader), accuracy

