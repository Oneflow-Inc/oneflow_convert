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

inceptionv3 = ModelCreator.create_model("inception_v3", pretrained=False)
inceptionv3.eval()

class inceptionv3Graph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = inceptionv3

    def build(self, x):
        out, aux = self.m(x)
        return out

def test_inceptionv3():
    
    inceptionv3_graph = inceptionv3Graph()
    inceptionv3_graph._compile(flow.randn(1, 3, 299, 299))

    convert_to_onnx_and_check(inceptionv3_graph, onnx_model_path=".", print_outlier=True)

test_inceptionv3()
