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
import tensorflow as tf

from oneflow_onnx.x2oneflow.util import load_tensorflow2_module_and_check

def test_concat():
    class Net(tf.keras.Model):
        def call(self, x):
            y = x * 3
            return tf.keras.layers.Concatenate()([x, y])

    load_tensorflow2_module_and_check(Net)


def test_concat_with_axis():
    class Net(tf.keras.Model):
        def call(self, x):
            y = x * 3
            return tf.keras.layers.Concatenate(axis=1)([x, y])

    load_tensorflow2_module_and_check(Net)


def test_unsqueeze():
    class Net(tf.keras.Model):
        def call(self, x):
            return tf.expand_dims(x, axis=2)

    load_tensorflow2_module_and_check(Net)


def test_transpose():
    class Net(tf.keras.Model):
        def call(self, x):
            # shape = x.shape
            return tf.transpose(x, perm=[0, 3, 1, 2])

    load_tensorflow2_module_and_check(Net)


def test_gather():
    class Net(tf.keras.Model):
        def call(self, x):
            return x[1]

    load_tensorflow2_module_and_check(Net)


def test_tensor_index():
    class Net(tf.keras.Model):
        def call(self, x):
            return x[0, 1:3, :1, 2:4]

    load_tensorflow2_module_and_check(Net)

