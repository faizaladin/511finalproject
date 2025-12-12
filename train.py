import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
from model import SqueezeNet 

# Config
BATCH_SIZE = 64     
NUM_EPOCHS = 100          
LEARNING_RATE = 0.04       
MOMENTUM = 0.9            
WEIGHT_DECAY = 5e-4         
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Image Resizing
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# Cifar10
print("Preparing Data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model Init
model = SqueezeNet(num_classes=10).to(device)

# Train Config
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# Train
sample_inputs, _ = next(iter(trainloader))
print(f"Sample batch image size: {sample_inputs.shape}")
wandb.init(project="squeezenet-cifar10", config={
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": "SGD",
    "lr_schedule": "linear"
})
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"Initial LR: {LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    # Accuracy
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    # Eval
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total

    wandb.log({
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "val_acc": val_acc,
        "lr": current_lr,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch+1} Summary:")
    print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    print(f"  Val Acc:    {val_acc:.2f}%")
    print(f"  Next LR:    {optimizer.param_groups[0]['lr']:.5f}")

    # Save checkpoint 
    torch.save(model.state_dict(), "squeezenet_cifar10.pth")

print("Training Finished. Model saved to squeezenet_cifar10.pth")