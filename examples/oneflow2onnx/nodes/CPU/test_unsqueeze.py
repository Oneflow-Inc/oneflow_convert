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

class Unsqueeze(flow.nn.Module):
    def __init__(self) -> None:
        super(Unsqueeze, self).__init__()
    
    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return flow.unsqueeze(x, dim=1)

unsqueeze = Unsqueeze()

class UnsqueezeOpGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = unsqueeze

    def build(self, x):
        return self.m(x)


def test_unsqueeze():
    
    unsqueeze_graph = UnsqueezeOpGraph()
    unsqueeze_graph._compile(flow.randn(1, 2, 3, 4))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(unsqueeze.state_dict(), tmpdirname)
        convert_to_onnx_and_check(unsqueeze_graph, flow_weight_dir=tmpdirname, onnx_model_path="/tmp", opset=11)

test_unsqueeze()
