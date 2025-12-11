import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SqueezeNet

# --- CONFIGURATION ---
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "squeezenet_cifar10.pth"

# --- DATA PREPARATION ---
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- MODEL LOADING ---
model = SqueezeNet(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- EVALUATION ---
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
