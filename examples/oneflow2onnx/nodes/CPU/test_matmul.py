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

class MatMul(flow.nn.Module):
    def __init__(self) -> None:
        super(MatMul, self).__init__()
        self.matmul = flow.nn.Linear(20, 30)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.matmul(x)

matmul = MatMul()
class matmulOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = matmul

    def build(self, x):
        out = self.m(x)
        return out


def test_matmul():
    
    matmul_graph = matmulOpGraph()
    matmul_graph._compile(flow.randn(1, 20))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(matmul.state_dict(), tmpdirname)
        convert_to_onnx_and_check(matmul_graph, flow_weight_dir=tmpdirname, onnx_model_path="/tmp")

test_matmul()
