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
import oneflow.typing as tp
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check


def test_flatten():
    @flow.global_function()
    def flatten(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        return flow.flatten(x, start_dim=1, end_dim=-1)

    convert_to_onnx_and_check(flatten)

def test_flatten_aixs_negative():
    @flow.global_function()
    def flatten(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        return flow.flatten(x, start_dim=0, end_dim=-1)

    convert_to_onnx_and_check(flatten)

def test_flatten_aixs_default():
    @flow.global_function()
    def flatten(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        return flow.flatten(x)

    convert_to_onnx_and_check(flatten)

def test_flatten_dtype_int():
    @flow.global_function()
    def flatten(x: tp.Numpy.Placeholder((3, 4, 2, 5))):
        x = flow.cast(x, flow.int32)
        return flow.flatten(x)

    convert_to_onnx_and_check(flatten, opset=11)
