import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import SqueezeNetCIFAR 

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
BATCH_SIZE = 256     # Typical for CIFAR-10
NUM_EPOCHS = 50        # Sufficient for convergence on CIFAR
LEARNING_RATE = 1e-4   # Adam optimizer starting learning rate

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- 2. DATA PREPARATION (CIFAR-10) ---
# Standard normalization for CIFAR-10
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. MODEL INITIALIZATION ---
# Using the SqueezeNetCIFAR class defined in the previous step
# Ensure you include the class definition from the previous response here
model = SqueezeNetCIFAR(num_classes=10).to(device)

# --- 4. OPTIMIZER & SCHEDULER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING LOOP ---
def train():
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1} Training")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate Epoch Metrics after epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.2f}%")

        # Validate every epoch (after epoch)
        validate()

        # Save model every epoch (after epoch)
        torch.save(model.state_dict(), f"squeezenet.pth")

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"--> Validation Accuracy: {acc:.2f}%")

# --- 6. EXECUTE ---
if __name__ == "__main__":
    train()