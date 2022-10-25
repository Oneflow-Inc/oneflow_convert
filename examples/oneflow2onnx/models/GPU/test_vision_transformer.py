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
import oneflow.nn as nn
from typing import Union, List, Dict, Any, cast
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

from flowvision.models import ModelCreator

import tempfile

vit = ModelCreator.create_model("vit_base_patch16_224", pretrained=False)
vit = vit.to("cuda")
vit.eval()


class VitGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = vit

    def build(self, x):
        out = self.m(x)
        return out


def test_vit():

    vit_graph = VitGraph()
    vit_graph._compile(flow.randn(1, 3, 224, 224).to("cuda"))

    convert_to_onnx_and_check(vit_graph, onnx_model_path=".", device="gpu")


test_vit()
