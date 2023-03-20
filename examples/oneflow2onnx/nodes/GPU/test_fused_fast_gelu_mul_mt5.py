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


class FusedFastGeluMT5(flow.nn.Module):
    def __init__(self) -> None:
        super(FusedFastGeluMT5, self).__init__()

    def forward(self, x: flow.Tensor, hidden: int) -> flow.Tensor:
        hidden_states = flow._C.fused_fast_gelu_mul(x, hidden)
        return hidden_states


fused_fast_gelu_mt5 = FusedFastGeluMT5()
fused_fast_gelu_mt5 = fused_fast_gelu_mt5.to("cuda")


class FusedFastGeluMT5OpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = fused_fast_gelu_mt5

    def build(self, x, hidden):
        hidden_states = self.m(x, hidden)
        return hidden_states


def test_fused_fast_gelu_mul_mt5():

    graph = FusedFastGeluMT5OpGraph()
    graph._compile(flow.randn(4, 3, 2).to("cuda"), flow.randn(4, 3, 2).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(fused_fast_gelu_mt5.state_dict(), tmpdirname, save_as_external_data=True)
        convert_to_onnx_and_check(graph, onnx_model_path="/tmp", device="gpu", input_tensor_range=[-0.001, 0.001])


test_fused_fast_gelu_mul_mt5()
