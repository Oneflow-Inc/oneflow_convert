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
import paddle.nn as nn
import paddle.nn.functional as F

from oneflow_onnx.x2oneflow.util import load_paddle_module_and_check


def test_concat():
    class Net(nn.Layer):
        def forward(self, x):
            y = x * 3
            return paddle.concat((x, y))

    load_paddle_module_and_check(Net)


def test_concat_with_axis():
    class Net(nn.Layer):
        def forward(self, x):
            y = x * 3
            return paddle.concat((x, y), axis=1)

    load_paddle_module_and_check(Net)


def test_unsqueeze():
    class Net(nn.Layer):
        def forward(self, x):
            return paddle.unsqueeze(x, 2)

    load_paddle_module_and_check(Net)


def test_transpose():
    class Net(nn.Layer):
        def forward(self, x):
            # shape = x.shape
            return paddle.transpose(x, perm=(0, 3, 1, 2))

    load_paddle_module_and_check(Net)


def test_gather():
    class Net(nn.Layer):
        def forward(self, x):
            return x[1]

    load_paddle_module_and_check(Net)


def test_tensor_index():
    class Net(nn.Layer):
        def forward(self, x):
            return x[0, 1:3, :1, 2:4]

    load_paddle_module_and_check(Net)

def test_split():
    class Net(nn.Layer):
        def forward(self, x):
            x1, y1 =  paddle.split(x,  num_or_sections=[3, 3], axis=1)
            return x1
    load_paddle_module_and_check(Net, input_size=(1, 6, 3, 3))
