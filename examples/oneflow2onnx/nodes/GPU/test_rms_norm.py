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


class RMSNorm(flow.nn.Module):
    def __init__(self) -> None:
        super(RMSNorm, self).__init__()
        self.norm = flow.nn.RMSNorm([10], eps=1e-6)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        y = self.norm(x)
        return y


layernorm = RMSNorm().to("cuda")
layernorm.eval()


class RMSNormOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = layernorm

    def build(self, x):
        out = self.m(x)
        return out


def test_rmsnorm():

    rms_norm_graph = RMSNormOpGraph()
    rms_norm_graph._compile(flow.randn(4, 5, 10).to("cuda"))

    convert_to_onnx_and_check(rms_norm_graph, onnx_model_path="/tmp", device="gpu")


test_rmsnorm()
