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
import tempfile
import random
import oneflow as flow
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

relu = flow.nn.ReLU()
relu = relu.to("cuda")


class ReLUOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = relu

    def build(self, x):
        out = self.m(x)
        return out


silu = flow.nn.SiLU()
silu = silu.to("cuda")


class SiLUOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = silu

    def build(self, x):
        out = self.m(x)
        return out


hard_swish = flow.nn.Hardswish()


class HardSwishOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = hard_swish

    def build(self, x):
        out = self.m(x)
        return out


hard_sigmoid = flow.nn.Hardsigmoid()


class HardSigmoidOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = hard_sigmoid

    def build(self, x):
        out = self.m(x)
        return out


prelu = flow.nn.PReLU()
prelu = prelu.to("cuda")


class PReLUOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = prelu

    def build(self, x):
        out = self.m(x)
        return out


gelu = flow.nn.GELU()
gelu = gelu.to("cuda")


class GeluOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = gelu

    def build(self, x):
        out = self.m(x)
        return out


new_gelu = flow.nn.GELU(approximate='tanh')
new_gelu = new_gelu.to("cuda")


class NewGeluOpGraph(flow.nn.Graph):
    def __init__(self) -> None:
        super().__init__()
        self.m = new_gelu

    def build(self, x):
        out = self.m(x)
        return out


quick_gelu = flow.nn.QuickGELU()
quick_gelu = quick_gelu.to("cuda")


class QuickGeluOpGraph(flow.nn.Graph):
    def __init__(self) -> None:
        super().__init__()
        self.m = quick_gelu

    def build(self, x):
        out = self.m(x)
        return out


def test_relu():

    relu_graph = ReLUOpGraph()
    relu_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(relu.state_dict(), tmpdirname)
        convert_to_onnx_and_check(relu_graph, onnx_model_path="/tmp", device="gpu")


def test_silu():

    silu_graph = SiLUOpGraph()
    silu_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    convert_to_onnx_and_check(silu_graph, onnx_model_path="/tmp", device="gpu")


def test_hard_swish():
    hard_swish_graph = HardSwishOpGraph()
    hard_swish_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(hard_swish.state_dict(), tmpdirname)
        convert_to_onnx_and_check(hard_swish_graph, onnx_model_path="/tmp", opset=14, device="gpu")


def test_hard_sigmoid():
    hard_sigmoid_graph = HardSigmoidOpGraph()
    hard_sigmoid_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(hard_swish.state_dict(), tmpdirname)
        convert_to_onnx_and_check(hard_sigmoid_graph, onnx_model_path="/tmp", device="gpu")


def test_prelu_one_channels():

    prelu_graph = PReLUOpGraph()
    prelu_graph._compile(flow.randn(1, 1, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(prelu.state_dict(), tmpdirname)
        convert_to_onnx_and_check(prelu_graph, onnx_model_path="/tmp", device="gpu")


def test_prelu_n_channels():

    prelu_graph = PReLUOpGraph()
    channels = random.randint(2, 10)
    prelu_graph._compile(flow.randn(1, channels, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(prelu.state_dict(), tmpdirname)
        convert_to_onnx_and_check(prelu_graph, onnx_model_path="/tmp", device="gpu")


def test_gelu():

    gelu_graph = GeluOpGraph()
    gelu_graph._compile(flow.randn(1, 3, 3).to("cuda"))

    convert_to_onnx_and_check(gelu_graph, onnx_model_path="/tmp", device="gpu")


def test_new_gelu():

    fast_gelu_graph = NewGeluOpGraph()
    fast_gelu_graph._compile(flow.randn(1, 3, 3).to("cuda"))

    convert_to_onnx_and_check(fast_gelu_graph, onnx_model_path="/tmp", device="gpu")


def test_quick_gelu():

    quick_gelu_graph = QuickGeluOpGraph()
    quick_gelu_graph._compile(flow.randn(1, 3, 3).to("cuda"))

    convert_to_onnx_and_check(quick_gelu_graph, onnx_model_path="/tmp", device="gpu")


test_prelu_one_channels()
test_prelu_n_channels()
test_relu()
test_silu()
test_hard_swish()
test_hard_sigmoid()
test_gelu()
test_new_gelu()
test_quick_gelu()
