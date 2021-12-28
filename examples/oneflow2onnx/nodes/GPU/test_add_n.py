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

class AddN(flow.nn.Module):
    def __init__(self) -> None:
        super(AddN, self).__init__()
    
    def forward(self, x: flow.Tensor) -> flow.Tensor:
        y = x + x + x
        return y

addn = AddN()
addn = addn.to("cuda")
class AddNOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = addn

    def build(self, x):
        out = self.m(x)
        return out


def test_addn():
    
    addn_graph = AddNOpGraph()
    addn_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))
    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(addn.state_dict(), tmpdirname)
        convert_to_onnx_and_check(addn_graph, flow_weight_dir=tmpdirname, onnx_model_path="/tmp", device="gpu")

test_addn()
