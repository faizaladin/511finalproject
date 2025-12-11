import torch
import torch.nn as nn
from torch.autograd import Function

class FixedPointQuantizeFunction(Function):
    @staticmethod
    def forward(ctx, x, total_bits, frac_bits):
        # 1. Calculate the resolution (step size)
        step = 2.0 ** (-frac_bits)
        
        # 2. Calculate the dynamic range
        # For signed 8-bit: -128 to 127 steps * step_size
        # Formula: Range = +/- 2^(total_bits - 1 - frac_bits)
        min_val = -(2.0 ** (total_bits - 1 - frac_bits))
        max_val = (2.0 ** (total_bits - 1 - frac_bits)) - step

        # 3. Clamp (Saturate) values that exceed the range
        x_clamped = torch.clamp(x, min_val, max_val)

        # 4. Quantize (Round to nearest step)
        x_quantized = torch.round(x_clamped / step) * step
        
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE):
        # Pass the gradient through unchanged so the model can learn
        return grad_output, None, None

class QuantizedConv2d(nn.Conv2d):
    """
    Drop-in replacement for nn.Conv2d that quantizes 
    both its Input (Activations) and Weights before convolving.
    """
    def __init__(self, *args, weight_frac_bits=6, input_frac_bits=4, **kwargs):
        super(QuantizedConv2d, self).__init__(*args, **kwargs)
        self.total_bits = 8
        self.w_frac = weight_frac_bits
        self.i_frac = input_frac_bits

    def forward(self, x):
        # 1. Quantize the Input (Activation from previous layer)
        x_q = FixedPointQuantizeFunction.apply(x, self.total_bits, self.i_frac)
        
        # 2. Quantize the Weights (on-the-fly)
        w_q = FixedPointQuantizeFunction.apply(self.weight, self.total_bits, self.w_frac)
        
        # 3. Standard Convolution using quantized values
        return self._conv_forward(x_q, w_q, self.bias)