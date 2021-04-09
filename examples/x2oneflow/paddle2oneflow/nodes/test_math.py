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
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from oneflow_onnx.x2oneflow.util import load_paddle_module_and_check


def test_add():
    class Net(nn.Layer):
        def forward(self, x):
            x += x
            return x

    load_paddle_module_and_check(Net)


def test_sub():
    class Net(nn.Layer):
        def forward(self, x):
            x -= 2
            return x

    load_paddle_module_and_check(Net)


def test_mul():
    class Net(nn.Layer):
        def forward(self, x):
            x *= x
            return x

    load_paddle_module_and_check(Net)


def test_div():
    class Net(nn.Layer):
        def forward(self, x):
            x /= 3
            return x

    load_paddle_module_and_check(Net)


def test_sqrt():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.sqrt(x)
            return x

    load_paddle_module_and_check(Net, input_min_val=0)


def test_pow():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.pow(x, 3)
            return x

    load_paddle_module_and_check(Net)


def test_tanh():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.tanh(x)
            return x

    load_paddle_module_and_check(Net)


def test_sigmoid():
    class Net(nn.Layer):
        def forward(self, x):
            m = nn.Sigmoid()
            x = m(x)
            return x

    load_paddle_module_and_check(Net)


# def test_erf():
#     class Net(nn.Layer):
#         def forward(self, x):
#             x = paddle.erf(x)
#             return x

#     load_paddle_module_and_check(Net)


def test_clip():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.clip(x, min=-1, max=2)
            return x

    load_paddle_module_and_check(Net)


# def test_cast():
#     class Net(nn.Layer):
#         def forward(self, x):
#             x = paddle.cast(x, 'float64')
#             return x

#     load_paddle_module_and_check(Net)

def test_abs():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.abs(x)
            return x

    load_paddle_module_and_check(Net)


def test_add_v2():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.add(x, x)
            return x

    load_paddle_module_and_check(Net)

def test_argmax():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.argmax(x, -1)
            return x

    load_paddle_module_and_check(Net)

def test_bmm():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.bmm(x, x)
            return x

    load_paddle_module_and_check(Net, input_size=(3, 2, 2))

def test_exp():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.exp(x)
            return x

    load_paddle_module_and_check(Net)

def test_floor():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.floor(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_hard_sigmoid():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.nn.functional. hardsigmoid(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_hard_swish():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.nn.functional.hardswish(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_leaky_relu():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.nn.functional.leaky_relu(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_log():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.log(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_matmul():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.matmul(x, x)
            return x
    
    load_paddle_module_and_check(Net, input_size=(3, 3))

def test_mean():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.mean(x, axis=-1)
            return x
    
    load_paddle_module_and_check(Net)

def test_prod():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.prod(x)
            return x
    
    load_paddle_module_and_check(Net, input_size=(3, ))

def test_scale():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.scale(x, scale=2.0, bias=1.0)
            return x
    
    load_paddle_module_and_check(Net)

def test_squeeze():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.squeeze(x, axis=1)
            return x
    
    load_paddle_module_and_check(Net, input_size=(5, 1, 10))

def test_sqrt():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.sqrt(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_square():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.square(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_stack():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.stack([x, x], axis=0)
            return x
    
    load_paddle_module_and_check(Net)

def test_stride_slice():
    class Net(nn.Layer):
        def forward(self, x):
            axes = [1, 2, 3]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            strides_1 = [1, 1, 1]
            strides_2 = [1, 1, 2]
            x = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)
            return x
    
    load_paddle_module_and_check(Net, input_size=(3, 4, 5, 6))

def test_swish():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.nn.functional.swish(x)
            return x
    
    load_paddle_module_and_check(Net)

def test_tanh():
    class Net(nn.Layer):
        def forward(self, x):
            x = paddle.tanh(x)
            return x
    
    load_paddle_module_and_check(Net)



