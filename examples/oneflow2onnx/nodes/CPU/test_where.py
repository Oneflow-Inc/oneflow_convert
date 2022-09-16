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


class WhereModule(flow.nn.Module):
    def __init__(self) -> None:
        super(WhereModule, self).__init__()
        self.y = flow.randn(1, 5)
        self.condition = flow.tensor([[True, True, False, True, False]], dtype=flow.bool)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return flow.where(self.condition, x, self.y)


m = WhereModule()


class whereOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = m

    def build(self, x):
        out = self.m(x)
        return out


def test_where():

    where_graph = whereOpGraph()
    where_graph._compile(flow.randn(1, 5))

    convert_to_onnx_and_check(where_graph, onnx_model_path="/tmp")


test_where()
