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


def test_add():
    class Net(tf.keras.Model):
        def call(self, x):
            x += x
            return x

    load_tensorflow2_module_and_check(Net)


def test_sub():
    class Net(tf.keras.Model):
        def call(self, x):
            x -= 2
            return x

    load_tensorflow2_module_and_check(Net)


def test_mul():
    class Net(tf.keras.Model):
        def call(self, x):
            x *= x
            return x

    load_tensorflow2_module_and_check(Net)


def test_div():
    class Net(tf.keras.Model):
        def call(self, x):
            x /= 3
            return x

    load_tensorflow2_module_and_check(Net)


def test_sqrt():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.math.sqrt(x)
            return x

    load_tensorflow2_module_and_check(Net, input_min_val=0)


def test_pow():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.math.pow(x, 3)
            return x

    load_tensorflow2_module_and_check(Net)


def test_tanh():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.keras.activations.tanh(x)
            return x

    load_tensorflow2_module_and_check(Net)


def test_sigmoid():
    class Net(tf.keras.Model):
        def call(self, x):
            m = tf.keras.activations.sigmoid(x)
            return x

    load_tensorflow2_module_and_check(Net)


def test_erf():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.math.erf(x)
            return x

    load_tensorflow2_module_and_check(Net)

# def test_cast():
#     class Net(tf.keras.Model):
#         def call(self, x):
#             x = tf.cast(x, tf.int32)
#             return x

#     load_tensorflow2_module_and_check(Net)

def test_abs():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.math.abs(x)
            return x
    
    load_tensorflow2_module_and_check(Net)

def test_exp():
    class Net(tf.keras.Model):
        def call(self, x):
            x = tf.math.exp(x)
            return x
    
    load_tensorflow2_module_and_check(Net)

def test_rsqrt():
    class Net(tf.keras.Model):
       def call(self, x):
           x = tf.math.rsqrt(x)
           return x

    load_tensorflow2_module_and_check(Net)

def test_maximum():
    class Net(tf.keras.Model):
       def call(self, x):
           x = tf.math.maximum(x, x*2)
           return x

    load_tensorflow2_module_and_check(Net)

def test_floordiv():
    class Net(tf.keras.Model):
       def call(self, x):
           x = tf.math.floordiv(x*1.5, x)
           return x

    load_tensorflow2_module_and_check(Net)
