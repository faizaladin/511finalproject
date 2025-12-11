import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model_qat import SqueezeNetQAT

# --- SETTINGS ---
TASK1_CHECKPOINT = "squeezenet_cifar10.pth" # Your file from Task 1
QAT_EPOCHS = 10         # Needs fewer epochs than fresh training
LR = 0.001              # Lower Learning Rate for fine-tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA ---
# Use same transforms as Task 1
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# --- LOAD MODEL ---
print("Initializing QAT Model...")
model = SqueezeNetQAT(num_classes=10).to(device)

print(f"Loading weights from {TASK1_CHECKPOINT}...")
# We can load the float weights directly because the parameter names match exactly
state_dict = torch.load(TASK1_CHECKPOINT, map_location=device)
model.load_state_dict(state_dict, strict=False)

# --- OPTIMIZER ---
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# --- QAT LOOP ---
print("Starting Quantization-Aware Training...")
for epoch in range(QAT_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{QAT_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # This uses QuantizedConv2d
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print(f"Epoch {epoch+1} Train Acc: {100.*correct/total:.2f}%")

    # Validation
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
            
    print(f"Epoch {epoch+1} Val Acc: {100.*val_correct/val_total:.2f}%")

# Save the final quantized model
torch.save(model.state_dict(), "squeezenet_qat_8bit.pth")
print("Saved QAT model to squeezenet_qat_8bit.pth")