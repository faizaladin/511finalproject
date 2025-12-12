import torch
import numpy as np
import matplotlib.pyplot as plt
from model_qat import SqueezeNetQAT, W_FRAC, A_FRAC

# Config
CHECKPOINT = "squeezenet_qat_8bit.pth" 
device = torch.device("cpu")

def quantize_tensor(x, total_bits=8, frac_bits=6):
    step = 2.0 ** (-frac_bits)
    min_val = -(2.0 ** (total_bits - 1 - frac_bits))
    max_val = (2.0 ** (total_bits - 1 - frac_bits)) - step
    x_clamped = torch.clamp(x, min_val, max_val)
    x_quantized = torch.round(x_clamped / step) * step
    return x_quantized

# Model Init
print(f"Loading {CHECKPOINT}...")
model = SqueezeNetQAT(num_classes=10).to(device)
try:
    state_dict = torch.load(CHECKPOINT, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

print("\n" + "="*40)
print("       VERIFICATION 1: WEIGHTS")
print("="*40)

raw_weight = model.features[0].weight.data
print(f"Layer: features[0] (Conv1)")
print(f"Configured W_FRAC: {W_FRAC} bits")
print(f"Expected Step Size: 2^-{W_FRAC} = {2**(-W_FRAC)}")

q_weight = quantize_tensor(raw_weight, total_bits=8, frac_bits=W_FRAC)

unique_vals = torch.unique(q_weight).numpy()
sorted_vals = np.sort(unique_vals)
diffs = np.diff(sorted_vals)
min_diff = diffs.min() if len(diffs) > 0 else 0

print(f"Unique discrete values found: {len(unique_vals)}")
print(f"Sample values: {sorted_vals[50:60]}") 
print(f"Measured Min Step Size: {min_diff:.6f}")

if np.isclose(min_diff, 2**(-W_FRAC), atol=1e-5):
    print("✅ SUCCESS: Weights are quantized correctly!")
else:
    print("❌ FAILURE: Weights do not match expected step size.")


print("\n" + "="*40)
print("     VERIFICATION 2: ACTIVATIONS")
print("="*40)
print(f"Configured A_FRAC: {A_FRAC} bits")
print(f"Expected Step Size: 2^-{A_FRAC} = {2**(-A_FRAC)}")


captured_act = []
def pre_hook_fn(module, input):
    captured_act.append(input[0].detach())


hook_handle = model.features[3].expand1x1.register_forward_pre_hook(pre_hook_fn)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
model(dummy_input)

act = captured_act[0]
unique_act = torch.unique(act).numpy()
sorted_act = np.sort(unique_act)
act_diffs = np.diff(sorted_act)
min_act_diff = act_diffs.min() if len(act_diffs) > 0 else 0

print(f"Layer: Fire2 Squeeze Output (Measured at Expand Input)")
print(f"Sample Output Values: {sorted_act[:10]} ...")
print(f"Measured Min Step Size: {min_act_diff:.6f}")

if np.isclose(min_act_diff, 2**(-A_FRAC), atol=1e-5):
    print("✅ SUCCESS: Activations are quantized correctly!")
else:
    print("❌ FAILURE: Activations do not match expected step size.")

hook_handle.remove()