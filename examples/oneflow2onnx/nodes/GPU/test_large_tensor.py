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


class LargeTensor(flow.nn.Module):
    def __init__(self) -> None:
        super(LargeTensor, self).__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return flow.ones((64, 1024, 1024), device="cuda") + flow.zeros((64, 1024, 1024), device="cuda") + x


large_tensor = LargeTensor()
large_tensor = large_tensor.to("cuda")


class LargeTensorOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = large_tensor

    def build(self, x):
        out = self.m(x)
        return out


def test_large_tensor():

    graph = LargeTensorOpGraph()
    graph._compile(flow.randn(1024, 1024).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(graph.state_dict(), tmpdirname, save_as_external_data=True)
        convert_to_onnx_and_check(graph, print_outlier=False, onnx_model_path="/tmp", device="gpu")


test_large_tensor()
