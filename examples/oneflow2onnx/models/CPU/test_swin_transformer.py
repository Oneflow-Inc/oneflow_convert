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

import oneflow as flow
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

from flowvision.models import ModelCreator

import tempfile

swin_transformer = ModelCreator.create_model("swin_base_patch4_window7_224", pretrained=False)
swin_transformer.eval()

class SwinTransformerGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = swin_transformer

    def build(self, x):
        out = self.m(x)
        return out

def test_swin_transformer():
    
    swin_transformer_graph = SwinTransformerGraph()
    swin_transformer_graph._compile(flow.randn(1, 3, 224, 224))
    # print(swin_transformer_graph._full_graph_proto)
    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(swin_transformer.state_dict(), tmpdirname)
        convert_to_onnx_and_check(swin_transformer_graph, flow_weight_dir=tmpdirname, onnx_model_path=".", print_outlier=False)

test_swin_transformer()
