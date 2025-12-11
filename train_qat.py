import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# Ensure quantization_utils.py is in the same folder
from model_qat import SqueezeNetQAT

# --- CONFIGURATION ---
TASK1_CHECKPOINT = "squeezenet_cifar10.pth" 
QAT_EPOCHS = 10         
LR = 1e-5               # FIX 1: Very low learning rate to prevent explosion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running QAT on: {device}")

# --- DATA ---
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
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# --- MODEL SETUP ---
model = SqueezeNetQAT(num_classes=10).to(device)

# --- STRICT LOADING ---
print(f"Loading weights from {TASK1_CHECKPOINT}...")
if not os.path.exists(TASK1_CHECKPOINT):
    raise FileNotFoundError(f"Could not find {TASK1_CHECKPOINT}. Please check the file name.")

state_dict = torch.load(TASK1_CHECKPOINT, map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

try:
    model.load_state_dict(new_state_dict, strict=True)
    print("SUCCESS: Weights loaded successfully.")
except RuntimeError as e:
    print("\nERROR: Model architecture mismatch.")
    print(e)
    exit(1)

# --- PRE-TRAINING CHECK ---
print("Running 'Epoch 0' verification...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if i > 10: break 
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

initial_acc = 100. * correct / total
print(f"Initial Accuracy (Quantized, Pre-tuning): {initial_acc:.2f}%")

if initial_acc < 20.0:
    print("\n[STOPPING] The model accuracy is too low (~10%). Weights are not loading correctly.")
    exit(1)
else:
    print("Model verified! Starting QAT fine-tuning...")

# --- OPTIMIZER ---
# FIX 2: Set weight_decay=0. We don't want to shrink weights that need to be large for quantization.
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0)
criterion = nn.CrossEntropyLoss()

# --- TRAINING LOOP ---
for epoch in range(QAT_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(trainloader, desc=f"QAT Epoch {epoch+1}/{QAT_EPOCHS}")
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        
        # FIX 3: Check for explosion before it breaks the model
        if torch.isnan(loss):
            print("ERROR: Loss is NaN! Gradients exploded.")
            break
            
        loss.backward()
        
        # FIX 4: Gradient Clipping (The Magic Fix)
        # This limits the update size even if the error is huge (127 vs 953)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

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
    
    torch.save(model.state_dict(), "squeezenet_qat_8bit.pth")

print("QAT Finished. Saved to squeezenet_qat_8bit.pth")