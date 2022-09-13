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


class Squeeze(flow.nn.Module):
    def __init__(self) -> None:
        super(Squeeze, self).__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return flow.squeeze(x, dim=1)


squeeze = Squeeze()
squeeze = squeeze.to("cuda")


class SqueezeOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = squeeze

    def build(self, x):
        return self.m(x)


def test_squeeze():

    squeeze_graph = SqueezeOpGraph()
    squeeze_graph._compile(flow.randn(2, 1, 2, 1, 2).to("cuda"))

    convert_to_onnx_and_check(squeeze_graph, onnx_model_path="/tmp", opset=11, device="gpu")


test_squeeze()
