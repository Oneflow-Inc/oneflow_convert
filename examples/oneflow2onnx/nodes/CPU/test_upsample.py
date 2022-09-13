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


class UpsampleNearest2D(flow.nn.Module):
    def __init__(self) -> None:
        super(UpsampleNearest2D, self).__init__()
        self.m = flow.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.m(x)


upsample_nearest_2d = UpsampleNearest2D()


class UpsampleNearest2DOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = upsample_nearest_2d

    def build(self, x):
        return self.m(x)


def test_upsample_nearest_2d():

    upsample_nearest2d_graph = UpsampleNearest2DOpGraph()
    upsample_nearest2d_graph._compile(flow.randn(1, 1, 2, 2))

    convert_to_onnx_and_check(upsample_nearest2d_graph, onnx_model_path="/tmp", opset=10)


test_upsample_nearest_2d()
