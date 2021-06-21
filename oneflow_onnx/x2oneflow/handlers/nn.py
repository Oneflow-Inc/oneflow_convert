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
import string
import random
import operator
from functools import reduce

import numpy as np

from oneflow_onnx.x2oneflow.handler import BackendHandler
from oneflow_onnx.x2oneflow.handler import flow_func
from oneflow_onnx.x2oneflow.handler import onnx_op
from oneflow_onnx.x2oneflow.handlers.common import ConvMixin
from oneflow_onnx.x2oneflow.handler import oneflow_code_gen, oneflow_blobname_map

@onnx_op("Conv")
class Conv(ConvMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.conv(node, tensor_dict)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls.conv(node, tensor_dict)


@onnx_op("BatchNormalization")
@flow_func(flow.layers.batch_normalization)
class BatchNormalization(BackendHandler):
    @classmethod
    def get_attrs_processor_param(cls):
        return {
            "default": {"epsilon": 1e-5},
        }

    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]

        # code gen for batchnorm
        func = 'weight_initializer = flow.truncated_normal(0.1)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        func = 'weight_regularizer = flow.regularizers.l2(0.0005)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        scale = tensor_dict[node.input_tensor_names[1]]
        offset = tensor_dict[node.input_tensor_names[2]]
        mean = tensor_dict[node.input_tensor_names[3]]
        variance = tensor_dict[node.input_tensor_names[4]]
        epsilon = node.attrs.get("epsilon", 1e-5)

        func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
        func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
        func = func + 'shape={}, '.format(list(scale.shape))
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        
        func = '{} = flow.get_variable('.format(node.input_tensor_names[2])
        func = func + 'name={}, '.format("'"+node.input_tensor_names[2]+"'")
        func = func + 'shape={}, '.format(list(offset.shape))
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        
        func = '{} = flow.get_variable('.format(node.input_tensor_names[3])
        func = func + 'name={}, '.format("'"+node.input_tensor_names[3]+"'")
        func = func + 'shape={}, '.format(list(mean.shape))
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        
        func = '{} = flow.get_variable('.format(node.input_tensor_names[4])
        func = func + 'name={}, '.format("'"+node.input_tensor_names[4]+"'")
        func = func + 'shape={}, '.format(list(variance.shape))
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        
        func = '{} = flow.nn.batch_normalization('.format(node.output_tensor_names[0])
        func = func + 'x={}, mean={}, variance={}, offset={}, scale={}, axis=1, variance_epsilon={})\n'.format(node.input_tensor_names[0], node.input_tensor_names[3],
                                                                                                        node.input_tensor_names[4], node.input_tensor_names[2], node.input_tensor_names[1], epsilon)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        y = flow.nn.batch_normalization(x, mean=mean, variance=variance, offset=offset, scale=scale, axis=1, variance_epsilon=epsilon)
        
        if y not in oneflow_blobname_map:
            oneflow_blobname_map[y] = node.output_tensor_names[0]

        return y

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


class PoolMixin(object):
    @classmethod
    def pool(cls, node, input_dict, pooling_type, strict=True):
        x = input_dict[node.input_tensor_names[0]]
        orig_x = x

        kernel_shape = node.attrs["kernel_shape"]

        spatial_size = len(kernel_shape)
        x_rank = spatial_size + 2

        kernel_shape = node.attrs["kernel_shape"]
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = node.attrs.get("ceil_mode")
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            pads = np.reshape(pads, [2, spatial_size]).T.tolist()
            pads = [[0, 0], [0, 0]] + pads

        # oneflow now not support ceil_mode pool, so this is a temporary solution
        if ceil_mode == 1:

            if (x.shape[2] + pads[2][0] + pads[2][1] - 1) % strides[0] != 0:
                pads[2][1] = pads[2][1] + (strides[0] - 1)

            if (x.shape[3] + pads[3][0] + pads[3][1] - 1) % strides[1] != 0:
                pads[3][1] = pads[3][1] + (strides[1] - 1)
        count_include_pad = bool(node.attrs.get("count_include_pad", 0))
        if count_include_pad != 0:
            x = flow.pad(
                x,
                paddings=(
                    (pads[0][0], pads[0][1]),
                    (pads[1][0], pads[1][1]),
                    (pads[2][0], pads[2][1]),
                    (pads[3][0], pads[3][1]),
                ),
            )
            func = '{} = flow.pad({}, paddings=(({}, {}), ({}, {}), ({}, {}), ({}, {})))\n'.format(node.input_tensor_names[0], node.input_tensor_names[0], pads[0][0],
                    pads[0][1], pads[1][0], pads[1][1], pads[2][0], pads[2][1], pads[3][0], pads[3][1])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            pads = [[0, 0], [0, 0], [0, 0], [0, 0]]
            # raise ValueError("count_include_pad != 0 is not supported")
        
        pool_type = ''

        if pooling_type == "AVG":
            op = flow.nn.avg_pool2d
            pool_type = 'flow.nn.avg_pool2d('
        elif pooling_type == "MAX":
            op = flow.nn.max_pool2d
            pool_type = 'flow.nn.max_pool2d('
        elif pooling_type == "MAX_WITH_ARGMAX":
            raise ValueError("maxpooling with argmax is not supported")

        if spatial_size != 2:
            raise ValueError("non-2d pooling is not supported")
        if node.attrs.get("storage_order", 0) != 0:
            raise ValueError("storage_order != 0 is not supported")

        # code gen for avgpool2d and maxpool2d pool
        oneflow_blobname_map[x] = node.input_tensor_names[0]
        
        func = '{} = '.format(node.output_tensor_names[0])
        func = func + pool_type
        func = func + node.input_tensor_names[0] + ', '
        func = func + 'ksize={}, '.format(kernel_shape)
        func = func + 'strides={}, '.format(strides)
        func = func + 'padding=(({}, {}), ({}, {}), ({}, {}), ({}, {})), '.format(pads[0][0],
                    pads[0][1], pads[1][0], pads[1][1], pads[2][0], pads[2][1], pads[3][0], pads[3][1])
        func = func + 'data_format={})\n'.format("'NCHW'")
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        y =  op(
            x, ksize=kernel_shape, strides=strides, padding=pads, data_format="NCHW"
        )
        if y not in oneflow_blobname_map:
            oneflow_blobname_map[y] = node.output_tensor_names[0]
        return y


@onnx_op("AveragePool")
class AveragePool(PoolMixin, BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.pool(node, tensor_dict, "AVG", kwargs.get("strict", True))

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("MaxPool")
class MaxPool(PoolMixin, BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        pool_type = "MAX" if len(node.output_tensor_names) == 1 else "MAX_WITH_ARGMAX"
        return cls.pool(node, tensor_dict, pool_type, kwargs.get("strict", True))

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Relu")
@flow_func(flow.math.relu)
class Relu(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_14(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Pad")
@flow_func(flow.pad)
class Pad(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        mode = node.attrs.pop("mode", "constant")
        if mode != "constant":
            raise NotImplementedError('Padding mode "{}" is not supported'.format(mode))

        if cls.SINCE_VERSION < 11:  # for opset 1 and opset 2
            node.attrs["paddings"] = node.attrs.pop("pads", None)
            node.attrs["constant_value"] = node.attrs.pop("value", 0.0)

        else:  # for opset 11
            init_dict = kwargs["init_dict"]
            paddings = (
                init_dict[node.input_tensor_names[1]]
                .reshape(2, -1)
                .transpose((1, 0))
                .tolist()
            )
            constant_values = (
                init_dict[node.input_tensor_names[2]].item()
                if len(node.input_tensor_names) == 3
                else 0
            )

        return [
            cls.run_onnx_node(
                node, tensor_dict, inputs=[x, paddings, constant_values], **kwargs
            )
        ]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_2(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("GlobalMaxPool")
class GlobalMaxPool(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        spatial_dims = list(range(2, len(x.shape)))
        return flow.math.reduce_max(x, spatial_dims, keepdims=True)


@onnx_op("GlobalAveragePool")
class GlobalAverageMaxPool(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        spatial_dims = list(range(2, len(x.shape)))
        func = '{} = flow.math.reduce_mean({}, axis={}, keepdims=True)\n'.format(node.output_tensor_names[0], node.input_tensor_names[0], spatial_dims)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        y = flow.math.reduce_mean(x, spatial_dims, keepdims=True)
        if y not in oneflow_blobname_map:
            oneflow_blobname_map[y] = node.output_tensor_names[0]
        return y


@onnx_op("Softmax")
class Softmax(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axis = node.attrs.get("axis", 1)
        axis = axis if axis >= 0 else len(np.shape(x)) + axis

        if x not in oneflow_blobname_map:
            oneflow_blobname_map[x] = node.input_tensor_names[0]

        if axis == len(np.shape(x)) - 1:
            func = '{} = flow.nn.softmax({})\n'.format(node.output_tensor_names[0], node.input_tensor_names[0])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            return flow.nn.softmax(x)

        shape = x.shape
        cal_shape = (
            reduce(operator.mul, shape[0:axis], 1),
            reduce(operator.mul, shape[axis : len(shape)], 1),
        )
        func = '{} = flow.reshape({}, {})\n'.format(node.input_tensor_names[0], node.input_tensor_names[0], cal_shape)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        func = '{} = flow.reshape(flow.nn.softmax({}), {})'.format(node.output_tensor_names[0], node.input_tensor_names[0], shape)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        x = flow.reshape(x, cal_shape)

        return flow.reshape(flow.nn.softmax(x), shape)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("LeakyRelu")
@flow_func(flow.nn.leaky_relu)
class LeakyRelu(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("PRelu")
@flow_func(flow.layers.prelu)
class PRelu(BackendHandler):

    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        name = node.input_tensor_names[0]

        cls.copy_variable_file(node.input_tensor_names[1], name + "-alpha")
        node.input_tensor_names = node.input_tensor_names[:1]

        return [
            cls.run_onnx_node(node, tensor_dict, name=name, **kwargs, attrs={"shared_axes": [2, 3]})
        ]

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)