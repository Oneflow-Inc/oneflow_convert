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
from oneflow import Tensor
import oneflow.nn as nn
from typing import Callable, Any, List
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

from flowvision.models import ModelCreator

import tempfile

shufflenet = ModelCreator.create_model("shufflenet_v2_x0_5", pretrained=False)
shufflenet = shufflenet.to("cuda")
shufflenet.eval()

class shufflenetGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = shufflenet

    def build(self, x):
        out = self.m(x)
        return out

def test_shufflenet():
    
    shufflenet_graph = shufflenetGraph()
    shufflenet_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        flow.save(shufflenet.state_dict(), tmpdirname)
        convert_to_onnx_and_check(shufflenet_graph, onnx_model_path=".", device="gpu")

test_shufflenet()
