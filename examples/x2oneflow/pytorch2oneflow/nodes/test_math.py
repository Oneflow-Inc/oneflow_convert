"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from oneflow_onnx.x2oneflow.util import load_pytorch_module_and_check


def test_add():
    class Net(nn.Module):
        def forward(self, x):
            x += x
            return x

    load_pytorch_module_and_check(Net)


def test_sub():
    class Net(nn.Module):
        def forward(self, x):
            x -= 2
            return x

    load_pytorch_module_and_check(Net)


def test_mul():
    class Net(nn.Module):
        def forward(self, x):
            x *= x
            return x

    load_pytorch_module_and_check(Net)


def test_div():
    class Net(nn.Module):
        def forward(self, x):
            x /= 3
            return x

    load_pytorch_module_and_check(Net)


def test_sqrt():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.sqrt(x)
            return x

    load_pytorch_module_and_check(Net, input_min_val=0)


def test_pow():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.pow(x, 3)
            return x

    load_pytorch_module_and_check(Net)


def test_tanh():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.tanh(x)
            return x

    load_pytorch_module_and_check(Net)


def test_sigmoid():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.sigmoid(x)
            return x

    load_pytorch_module_and_check(Net)


def test_erf():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.erf(x)
            return x

    load_pytorch_module_and_check(Net)


def test_clip():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.clamp(x, -1, 2)
            return x

    load_pytorch_module_and_check(Net)


# def test_cast():
#     class Net(nn.Module):
#         def forward(self, x):
#             x = x.int()
#             return x

#     load_pytorch_module_and_check(Net)

def test_abs():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.abs(x)
            return x
    
    load_pytorch_module_and_check(Net)

def test_acos():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.acos(x)
            return x
    
    load_pytorch_module_and_check(Net)

def test_add_v2():
    class Net(nn.Module):
        def forward(self, x):
            x = torch.add(x, 0.5)
            return x
    
    load_pytorch_module_and_check(Net)

def test_addmm():
    class Net(nn.Module):
        def forward(self, x):
            return torch.addmm(x, x, x)
    
    load_pytorch_module_and_check(Net, input_size=(2, 2))

def test_arange():
    class Net(nn.Module):
        def forward(self, x):
            return torch.arange(5)
    
    load_pytorch_module_and_check(Net)

def test_argmax():
    class Net(nn.Module):
        def forward(self, x):
            return torch.argmax(x)
    
    load_pytorch_module_and_check(Net)

def test_argmin():
    class Net(nn.Module):
        def forward(self, x):
            return torch.argmin(x)
    
    load_pytorch_module_and_check(Net)

def test_asin():
    class Net(nn.Module):
        def forward(self, x):
            return torch.asin(x)
    
    load_pytorch_module_and_check(Net)

def test_atan():
    class Net(nn.Module):
        def forward(self, x):
            return torch.atan(x)
    
    load_pytorch_module_and_check(Net)

def test_baddbmm():
    class Net(nn.Module):
        def forward(self, x):
            return torch.baddbmm(x, x, x)
    
    load_pytorch_module_and_check(Net, input_size=(2, 2, 2))

def test_and():
    class Net(nn.Module):
        def forward(self, x):
            return torch.baddbmm(x, x, x)
    
    load_pytorch_module_and_check(Net, input_size=(2, 2, 2))

def test_ceil():
    class Net(nn.Module):
        def forward(self, x):
            return torch.ceil(x)
    
    load_pytorch_module_and_check(Net)

def test_cos():
    class Net(nn.Module):
        def forward(self, x):
            return torch.cos(x)
    
    load_pytorch_module_and_check(Net)

def test_elu():
    class Net(nn.Module):
        def forward(self, x):
            m = nn.ELU()
            return m(x)
    
    load_pytorch_module_and_check(Net)

def test_eq():
    class Net(nn.Module):
        def forward(self, x):
            return torch.eq(x, x)
    
    load_pytorch_module_and_check(Net)

def test_exp():
    class Net(nn.Module):
        def forward(self, x):
            return torch.exp(x)
    
    load_pytorch_module_and_check(Net)

def test_floor():
    class Net(nn.Module):
        def forward(self, x):
            return torch.floor(x)
    
    load_pytorch_module_and_check(Net)


def test_floor_divide():
    class Net(nn.Module):
        def forward(self, x):
            a = torch.tensor([4.0, 3.0])
            b = torch.tensor([2.0, 2.0])
            return torch.floor_divide(a, b)
    
    load_pytorch_module_and_check(Net)

def test_full():
    class Net(nn.Module):
        def forward(self, x):
            return torch.full((2, 3), 1.5)
    
    load_pytorch_module_and_check(Net)

def test_full_like():
    class Net(nn.Module):
        def forward(self, x):
            return torch.full_like(x, 1.5)
    
    load_pytorch_module_and_check(Net)

def test_gelu():
    class Net(nn.Module):
        def forward(self, x):
            m = nn.GELU()
            return m(x)
    
    load_pytorch_module_and_check(Net)

def test_hardtanh():
    class Net(nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardtanh(x)
    
    load_pytorch_module_and_check(Net)

def test_leaky_relu():
    class Net(nn.Module):
        def forward(self, x):
            return torch.nn.functional.leaky_relu(x)
    
    load_pytorch_module_and_check(Net)

def test_log():
    class Net(nn.Module):
        def forward(self, x):
            return torch.log(x)
    
    load_pytorch_module_and_check(Net)

def test_log1p():
    class Net(nn.Module):
        def forward(self, x):
            return torch.log1p(x)
    
    load_pytorch_module_and_check(Net)

def test_log2():
    class Net(nn.Module):
        def forward(self, x):
            return torch.log2(x)
    
    load_pytorch_module_and_check(Net)

def test_log_softmax():
    class Net(nn.Module):
        def forward(self, x):
            return torch.nn.functional.log_softmax(x)
    
    load_pytorch_module_and_check(Net)

def test_logsumexp():
    class Net(nn.Module):
        def forward(self, x):
            return torch.logsumexp(x, dim=1)
    
    load_pytorch_module_and_check(Net)

def test_max():
    class Net(nn.Module):
        def forward(self, x):
            return torch.max(x)
    
    load_pytorch_module_and_check(Net)

def test_min():
    class Net(nn.Module):
        def forward(self, x):
            return torch.min(x)
    
    load_pytorch_module_and_check(Net)

def test_mean():
    class Net(nn.Module):
        def forward(self, x):
            return torch.mean(x)
    
    load_pytorch_module_and_check(Net)

def test_mm():
    class Net(nn.Module):
        def forward(self, x):
            return torch.mm(x, x)
    
    load_pytorch_module_and_check(Net, input_size=(2, 2))

def test_neg():
    class Net(nn.Module):
        def forward(self, x):
            return torch.neg(x)
    
    load_pytorch_module_and_check(Net)

def test_norm():
    class Net(nn.Module):
        def forward(self, x):
            return torch.norm(x)
    
    load_pytorch_module_and_check(Net)

def test_permute():
    class Net(nn.Module):
        def forward(self, x):
            x = x.permute(2, 0, 1)
            return x
    load_pytorch_module_and_check(Net, input_size=(2, 3, 5))

def test_prod():
    class Net(nn.Module):
        def forward(self, x):
            return torch.prod(x)
    
    load_pytorch_module_and_check(Net, input_size=(3, ))

def test_reshape_as():
    class Net(nn.Module):
        def forward(self, x):
            return x.reshape_as(x)
    load_pytorch_module_and_check(Net, input_size=(2, 3, 5))

def test_round():
    class Net(nn.Module):
        def forward(self, x):
            return torch.round(x)
    
    load_pytorch_module_and_check(Net)

def test_rsqrt():
    class Net(nn.Module):
        def forward(self, x):
            return torch.rsqrt(x)
    
    load_pytorch_module_and_check(Net)

def test_rsqrt():
    class Net(nn.Module):
        def forward(self, x):
            return torch.rsqrt(x)
    
    load_pytorch_module_and_check(Net)

def test_sign():
    class Net(nn.Module):
        def forward(self, x):
            return torch.sign(x)
    
    load_pytorch_module_and_check(Net)

def test_sin():
    class Net(nn.Module):
        def forward(self, x):
            return torch.sin(x)
    
    load_pytorch_module_and_check(Net)

def test_softplus():
    class Net(nn.Module):
        def forward(self, x):
            m = nn.Softplus()
            return m(x)
    
    load_pytorch_module_and_check(Net)

def test_squeeze():
    class Net(nn.Module):
        def forward(self, x):
            return torch.squeeze(x)
    
    load_pytorch_module_and_check(Net, input_size=(2, 1, 2, 1))

def test_tan():
    class Net(nn.Module):
        def forward(self, x):
            return torch.tan(x)
    
    load_pytorch_module_and_check(Net)

def test_tanh():
    class Net(nn.Module):
        def forward(self, x):
            return torch.tanh(x)
    
    load_pytorch_module_and_check(Net)
