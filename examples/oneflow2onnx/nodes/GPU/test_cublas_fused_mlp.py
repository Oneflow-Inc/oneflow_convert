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


class MLP(flow.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.mlp = flow.nn.FusedMLP(in_features=8, hidden_features=[16, 32], out_features=16, skip_final_activation=True)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.mlp(x)


mlp = MLP()
mlp = mlp.to("cuda")


class TestGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = mlp

    def build(self, x):
        out = self.m(x)
        return out


def test_cublas_fused_mlp():

    graph = TestGraph()
    graph._compile(flow.randn(32, 8).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(mlp.state_dict(), tmpdirname)
        convert_to_onnx_and_check(graph, onnx_model_path="/tmp", device="gpu")


test_cublas_fused_mlp()
