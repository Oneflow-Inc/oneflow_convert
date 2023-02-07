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
import oneflow as flow
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check


class Conv1d(flow.nn.Module):
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        self.conv = flow.nn.Conv1d(3, 16, 3, stride=2)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.conv(x)


conv1d_module = Conv1d()
conv1d_module = conv1d_module.to("cuda")


class Conv1dOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = conv1d_module

    def build(self, x):
        out = self.m(x)
        return out


def test_conv1d():

    conv1d_graph = Conv1dOpGraph()
    conv1d_graph._compile(flow.randn(1, 3, 224).to("cuda"))

    convert_to_onnx_and_check(conv1d_graph, onnx_model_path="/tmp", device="gpu")


def test_conv1d_opset14():

    conv1d_graph = Conv1dOpGraph()
    conv1d_graph._compile(flow.randn(1, 3, 224).to("cuda"))

    convert_to_onnx_and_check(conv1d_graph, onnx_model_path="/tmp", opset=14, device="gpu")


test_conv1d()
test_conv1d_opset14()


class Conv2d(flow.nn.Module):
    def __init__(self) -> None:
        super(Conv2d, self).__init__()
        self.conv = flow.nn.Conv2d(3, 16, 3)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.conv(x)


conv_module = Conv2d()
conv_module = conv_module.to("cuda")


class Conv2dOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = conv_module

    def build(self, x):
        out = self.m(x)
        return out


def test_conv2d():

    conv_graph = Conv2dOpGraph()
    conv_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(conv_module.state_dict(), tmpdirname, save_as_external_data=True)
        convert_to_onnx_and_check(conv_graph, onnx_model_path="/tmp", device="gpu")


test_conv2d()
