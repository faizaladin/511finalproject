import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from model import SqueezeNetCIFAR
import time
import warnings
from tqdm import tqdm
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import torch
torch.backends.quantized.engine = 'x86'
print(f"Quantization engine: {torch.backends.quantized.engine}")

# Config
BATCH_SIZE = 128
device = torch.device('cpu')

def get_dataloaders():
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return trainloader, testloader

def evaluate_model(model, loader, device_to_run=device, desc="Evaluating"):
    model.to(device_to_run)
    model.eval()
    
    correct = 0
    total = 0
    
    print(f"[{desc}] Inference Device: {device_to_run}")

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc, leave=False):
            inputs, targets = inputs.to(device_to_run), targets.to(device_to_run)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    
    accuracy = 100. * correct / total
    inference_time = (end_time - start_time) / len(loader) * 1000 # ms per batch
    return accuracy, inference_time

def get_model_size_mb(model):

    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size

def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully.")
    return model

def main():
    def clip_weights_to_int8(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.clamp_(-128, 127)
    trainloader, testloader = get_dataloaders()

    print("\n" + "="*40)
    print("1. Evaluating Baseline FP32 Model")
    print("="*40)
    model_fp32 = SqueezeNetCIFAR(num_classes=10)
    model_fp32 = load_checkpoint(model_fp32, 'squeezenet.pth')
    acc_fp32, time_fp32 = evaluate_model(model_fp32, testloader, device_to_run=torch.device('cpu'))
    size_fp32 = get_model_size_mb(model_fp32)
    print("-" * 40)
    print(f"FP32 Accuracy: {acc_fp32:.2f}%")
    print(f"FP32 Size: {size_fp32:.2f} MB")
    print(f"FP32 Inference Time (Batch): {time_fp32:.2f} ms")
    print("-" * 40)

    print("\n" + "="*40)
    print("2. Evaluating FX Graph Mode INT8 Quantized Model (x86)")
    print("="*40)

    torch.backends.quantized.engine = 'x86'
    from torch.ao.quantization import QConfigMapping, get_default_qconfig
    # # Use symmetric 8-bit quantization for both weights and activations
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig('x86'))
    # # Use a batch from testloader for calibration
    example_input, _ = next(iter(testloader))
    example_input = example_input.cpu()
    # Calibration: run a few batches through the model
    # Train FP32 model for 3 epochs with 8-bit clipping
    model_qat = SqueezeNetCIFAR(num_classes=10).train()
    model_qat = load_checkpoint(model_qat, 'squeezenet.pth')
    model_qat = model_qat.to('cuda')
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(3):
        running_loss = 0.0
        with tqdm(trainloader, desc=f"[QAT-style] Epoch {epoch+1}/3", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                optimizer.zero_grad()
                outputs = model_qat(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss/(pbar.n+1))
        clip_weights_to_int8(model_qat)
        print(f"[QAT-style] Epoch {epoch+1}/3, Loss: {running_loss/len(trainloader):.4f}")
    # Move model to CPU for quantization
    model_qat = model_qat.to('cpu')
    model_qat.eval()
    prepared_model_qat = prepare_fx(model_qat, qconfig_mapping, example_inputs=example_input)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(testloader):
            prepared_model_qat(inputs.cpu())
            if i >= 10:
                break
    qat_quantized = convert_fx(prepared_model_qat)
    acc_qat_int8, time_qat_int8 = evaluate_model(qat_quantized, testloader, device_to_run=torch.device('cpu'), desc="FX Quantized INT8 (x86) - QAT-style 3 epochs")
    size_qat_int8 = get_model_size_mb(qat_quantized)
    print("-" * 40)
    print(f"QAT-style INT8 Accuracy: {acc_qat_int8:.2f}%")
    print(f"QAT-style INT8 Size: {size_qat_int8:.2f} MB")
    print(f"QAT-style INT8 Inference Time (Batch): {time_qat_int8:.2f} ms")
    print("-" * 40)

if __name__ == "__main__":
    main()