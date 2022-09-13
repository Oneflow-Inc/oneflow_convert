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


class GroupConv2d(flow.nn.Module):
    def __init__(self) -> None:
        super(GroupConv2d, self).__init__()
        self.group_conv2d = flow.nn.Conv2d(16, 16, 3, groups=16)

    def forward(self, x: flow.Tensor):
        return self.group_conv2d(x)


group_conv_module = GroupConv2d()


class GraphConv2dOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = group_conv_module

    def build(self, x):
        out = self.m(x)
        return out


def test_group_conv2d():

    group_conv_graph = GraphConv2dOpGraph()
    group_conv_graph._compile(flow.randn(1, 16, 224, 224))

    convert_to_onnx_and_check(group_conv_graph, onnx_model_path="/tmp")


test_group_conv2d()
