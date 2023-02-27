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


class FusedBiasAddScaleMaskSoftmaxDropout(flow.nn.Module):
    def __init__(self) -> None:
        super(FusedBiasAddScaleMaskSoftmaxDropout, self).__init__()

    def forward(self, x: flow.Tensor, bias: flow.Tensor, mask: flow.Tensor, fill_value: int, scale: float) -> flow.Tensor:
        (output, softmax_output) = flow._C.fused_bias_add_scale_mask_softmax_dropout(x, bias, mask, fill_value=fill_value, scale=scale)
        return output, softmax_output


fused_bias_add_scale_mask_softmax_dropout = FusedBiasAddScaleMaskSoftmaxDropout()
fused_bias_add_scale_mask_softmax_dropout = fused_bias_add_scale_mask_softmax_dropout.to("cuda")


class FusedBiasAddScaleMaskSoftmaxDropoutOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = fused_bias_add_scale_mask_softmax_dropout

    def build(self, x, bias, mask, fill_value=-100, scale=1):
        output, softmax_output = self.m(x, bias, mask, fill_value, scale)
        return output, softmax_output


def test_fused_bias_add_scale_mask_softmax_dropout():

    graph = FusedBiasAddScaleMaskSoftmaxDropoutOpGraph()
    graph._compile(flow.randn(4, 2, 3).to("cuda"), flow.randn(4, 2, 3).to("cuda"), flow.randint(0, 2, size=[4, 2, 3]).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(fused_bias_add_scale_mask_softmax_dropout.state_dict(), tmpdirname, save_as_external_data=True)
        convert_to_onnx_and_check(graph, onnx_model_path="/tmp", device="gpu", input_tensor_range=[-0.001, 0.001])


test_fused_bias_add_scale_mask_softmax_dropout()