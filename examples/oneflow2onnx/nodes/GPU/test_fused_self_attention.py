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


class FusedSelfAtt(flow.nn.Module):
    def __init__(self) -> None:
        super(FusedSelfAtt, self).__init__()

    def forward(self, x: flow.Tensor, head_size: int, alpha: float) -> flow.Tensor:
        (fused_qmk, fused_v) = flow._C.fused_self_attention(x, head_size=head_size, alpha=alpha,)
        return fused_qmk, fused_v

fused_self_att = FusedSelfAtt()
fused_self_att = fused_self_att.to("cuda")


class TestGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = fused_self_att

    def build(self, x, head_size=64, alpha=1):
        fused_qmk, fused_v = self.m(x, head_size, alpha)
        return fused_qmk, fused_v


def test_fused_self_attention():

    graph = TestGraph()
    graph._compile(flow.randn(512, 4, 768).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(fused_self_att.state_dict(), tmpdirname)
        convert_to_onnx_and_check(graph, onnx_model_path="/tmp", device="gpu")


test_fused_self_attention()
