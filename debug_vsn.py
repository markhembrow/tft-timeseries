"""Test VSN weight_ctx bias gradient in isolation."""
import sys, os, torch, torch.nn as nn
from torch.nn.functional import silu
sys.path.insert(0, '/home/mctouch/code/tft-timeseries')

from tft_timeseries.model import GatedResidualNetwork, VariableSelectionNetwork

B, S, D_in, D_mod = 4, 3, 1, 32
xi = torch.randn(B, S, D_in)
vsn = VariableSelectionNetwork(num_inputs=S, d_model=D_mod, d_hidden=D_mod*4, dropout=0.0)

out = vsn(xi)
print("out shape", out.shape)
loss = out.sum()
loss.backward()
print("weight_ctx bias grad:", vsn.weight_ctx.bias.grad)
print("weight_ctx weight grad:", vsn.weight_ctx.weight.grad)
