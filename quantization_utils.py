import torch
import torch.nn as nn
from torch.autograd import Function

class FixedPointQuantizeFunction(Function):
    @staticmethod
    def forward(ctx, x, total_bits, frac_bits):
        step = 2.0 ** (-frac_bits)
        min_val = -(2.0 ** (total_bits - 1 - frac_bits))
        max_val = (2.0 ** (total_bits - 1 - frac_bits)) - step
        x_clamped = torch.clamp(x, min_val, max_val)
        x_quantized = torch.round(x_clamped / step) * step
        
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class QuantizedConv2d(nn.Conv2d):
    """quantizes both its Input (Activations) and Weights before convolving."""

    def __init__(self, *args, weight_frac_bits=6, input_frac_bits=4, **kwargs):
        super(QuantizedConv2d, self).__init__(*args, **kwargs)
        self.total_bits = 8
        self.w_frac = weight_frac_bits
        self.i_frac = input_frac_bits

    def forward(self, x):
        # Quantize the Input 
        x_q = FixedPointQuantizeFunction.apply(x, self.total_bits, self.i_frac)
        
        # Quantize the Weights
        w_q = FixedPointQuantizeFunction.apply(self.weight, self.total_bits, self.w_frac)
        
        # Convolution using quantized values
        return self._conv_forward(x_q, w_q, self.bias)