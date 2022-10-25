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


class FlattenTranspose(flow.nn.Module):
    def __init__(self) -> None:
        super(FlattenTranspose, self).__init__()

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        res = x.flatten(2).transpose(1, 2)
        print(res.shape)
        return res

flatten_transpose = FlattenTranspose()
flatten_transpose = flatten_transpose.to("cuda")


class FlattenTransposeOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = flatten_transpose

    def build(self, x):
        out = self.m(x)
        return out


def test_flatten_transpose():

    flatten_transpose_graph = FlattenTransposeOpGraph()
    flatten_transpose_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(flatten_transpose_graph.state_dict(), tmpdirname)
        convert_to_onnx_and_check(flatten_transpose_graph, onnx_model_path="/tmp", device="gpu")


test_flatten_transpose()
