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


class MathOps(flow.nn.Module):
    def __init__(self) -> None:
        super(MathOps, self).__init__()
        self.A = flow.randn(128, 224)
        self.indices = flow.tensor([0,2])

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = x / 1
        y1 = x * x
        y2 = y1 / x
        y2 = y2 - x
        y2 = y2 + x
        y2 = flow.abs(y2)
        y2 = flow.ceil(y2)
        y3 = flow.clip(x, -1.0, 1.0)
        y3 = flow.acos(y3)
        y3 = flow.pow(y3, 2.0)
        y2 = y2 + y3

        B = x # shape: (1, 3, 224, 224)
        #flow.matmul(self.A, B)
        flow.lt(x, 1)
        flow.gt(x, 1)
        #flow._C.gather(x, self.indices, 1)
        return y2


math_ops = MathOps()


class MathOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = math_ops

    def build(self, x):
        out = self.m(x)
        return out


def test_math_ops():

    math_ops_graph = MathOpGraph()
    math_ops_graph._compile(flow.randn(1, 3, 224, 224))

    convert_to_onnx_and_check(math_ops_graph, onnx_model_path="/tmp")


test_math_ops()
