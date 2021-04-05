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
