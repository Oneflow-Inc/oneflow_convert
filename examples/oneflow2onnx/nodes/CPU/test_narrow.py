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


class Narrow(flow.nn.Module):
    def __init__(self) -> None:
        super(Narrow, self).__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return flow.narrow(x, 0, 0, 2)


narrow = Narrow()


class NarrowOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = narrow

    def build(self, x):
        return self.m(x)


def test_narrow():

    narrow_graph = NarrowOpGraph()
    narrow_graph._compile(flow.randn(3, 3))

    convert_to_onnx_and_check(narrow_graph, onnx_model_path="/tmp", opset=11)


test_narrow()
