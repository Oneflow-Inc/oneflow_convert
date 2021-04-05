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


def generate_reduction_test(flow_op, *args, **kwargs):
    @flow.global_function()
    def job(x: tp.Numpy.Placeholder((3, 5, 4))):
        return flow_op(x, *args, **kwargs)

    convert_to_onnx_and_check(job)


def test_reduce_sum():
    generate_reduction_test(flow.math.reduce_sum)


def test_reduce_sum_axis_12():
    generate_reduction_test(flow.math.reduce_sum, axis=[1, 2])


def test_reduce_sum_axis_12_keepdim():
    generate_reduction_test(flow.math.reduce_sum, axis=[1, 2], keepdims=True)


def test_reduce_prod():
    generate_reduction_test(flow.math.reduce_prod)


def test_reduce_prod_axis_12():
    generate_reduction_test(flow.math.reduce_prod, axis=[1, 2])


def test_reduce_prod_axis_12_keepdim():
    generate_reduction_test(flow.math.reduce_prod, axis=[1, 2], keepdims=True)


def test_reduce_mean():
    generate_reduction_test(flow.math.reduce_mean)


def test_reduce_mean_axis_12():
    generate_reduction_test(flow.math.reduce_mean, axis=[1, 2])


def test_reduce_mean_axis_12_keepdim():
    generate_reduction_test(flow.math.reduce_mean, axis=[1, 2], keepdims=True)


def test_reduce_min():
    generate_reduction_test(flow.math.reduce_min)


def test_reduce_min_axis_12():
    generate_reduction_test(flow.math.reduce_min, axis=[1, 2])


def test_reduce_min_axis_12_keepdim():
    generate_reduction_test(flow.math.reduce_min, axis=[1, 2], keepdims=True)
