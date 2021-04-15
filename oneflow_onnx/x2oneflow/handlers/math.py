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
from oneflow_onnx.x2oneflow.handler import BackendHandler
from oneflow_onnx.x2oneflow.handler import onnx_op
from oneflow_onnx.x2oneflow.handler import flow_func
from oneflow_onnx.x2oneflow.handlers.common import ArithmeticMixin, BasicMathMixin
from oneflow_onnx import util as onnx_util
from oneflow_onnx.x2oneflow.handler import oneflow_code_gen, oneflow_blobname_map

@onnx_op("Add")
@flow_func(flow.math.add)
class Add(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        if tensor_dict[node.input_tensor_names[1]] not in oneflow_blobname_map:
            oneflow_blobname_map[tensor_dict[node.input_tensor_names[1]]] = node.input_tensor_names[1]
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
            func = func + 'shape={}, '.format(list(tensor_dict[node.input_tensor_names[1]].shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Sub")
@flow_func(flow.math.subtract)
class Sub(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        if tensor_dict[node.input_tensor_names[1]] not in oneflow_blobname_map:
            oneflow_blobname_map[tensor_dict[node.input_tensor_names[1]]] = node.input_tensor_names[1]
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
            func = func + 'shape={}, '.format(list(tensor_dict[node.input_tensor_names[1]].shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Mul")
@flow_func(flow.math.multiply)
class Mul(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        if tensor_dict[node.input_tensor_names[1]] not in oneflow_blobname_map:
            # code gen for conv weight_initializer
            func = 'weight_initializer = flow.truncated_normal(0.1)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            #code gen for conv weight_regularizer
            func = 'weight_regularizer = flow.regularizers.l2(0.0005)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)

            oneflow_blobname_map[tensor_dict[node.input_tensor_names[1]]] = node.input_tensor_names[1]
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
            func = func + 'shape={}, '.format(list(tensor_dict[node.input_tensor_names[1]].shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Div")
@flow_func(flow.math.divide)
class Div(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.limited_broadcast(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        if tensor_dict[node.input_tensor_names[1]] not in oneflow_blobname_map:
            oneflow_blobname_map[tensor_dict[node.input_tensor_names[1]]] = node.input_tensor_names[1]
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
            func = func + 'shape={}, '.format(list(tensor_dict[node.input_tensor_names[1]].shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Pow")
@flow_func(flow.math.pow)
class Pow(ArithmeticMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]
        if y not in oneflow_blobname_map:
            func = 'weight_initializer = flow.truncated_normal(0.1)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            func = 'weight_regularizer = flow.regularizers.l2(0.0005)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
            func = func + 'shape={}, '.format(list(y.shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)

        if len(y.shape) > len(x.shape):
            x = flow.math.broadcast_to_compatible_with(x, [y])
            func = '{} = flow.math.broadcast_to_compatible_with({}, [{}])\n'.format(node.input_tensor_names[0], node.input_tensor_names[0], node.input_tensor_names[1])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        elif len(x.shape) > len(y.shape):
            func = '{} = flow.math.broadcast_to_compatible_with({}, [{}])\n'.format(node.input_tensor_names[1], node.input_tensor_names[1], node.input_tensor_names[0])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            y = flow.math.broadcast_to_compatible_with(y, [x])
        
        if y not in oneflow_blobname_map:
            oneflow_blobname_map[y] = node.input_tensor_names[1]
        
        func = '{} = flow.math.pow({}, {})\n'.format(node.output_tensor_names[0], node.input_tensor_names[0], node.input_tensor_names[1])
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        z = flow.math.pow(x, y)
        if z not in oneflow_blobname_map:
            oneflow_blobname_map[z] = node.output_tensor_names[0]
        return z

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls.version_1(node, tensor_dict, **kwargs)


@onnx_op("Tanh")
@flow_func(flow.math.tanh_v2)
class Tanh(BasicMathMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return [cls.run_onnx_node(node, tensor_dict, **kwargs)]


@onnx_op("Sigmoid")
@flow_func(flow.math.sigmoid)
class Sigmoid(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("HardSigmoid")
class HardSigmoid(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        alpha = node.attrs.get("alpha")
        beta = node.attrs.get("beta")
        return flow.clip(x * alpha + beta, 0, 1.0)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]

        # code gen for gemm B
        gemm_weight_shape = list(tensor_dict[node.input_tensor_names[1]].shape)
        # code gen for gemm weight_initializer
        func = 'gemm_initializer = flow.truncated_normal(0.1)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        #code gen for gemm weight_regularizer
        func = 'gemm_regularizer = flow.regularizers.l2(0.0005)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        # code gen for gemm weight_shape
        # code gen for gemm weights
        func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
        func = func + 'name={}, '.format("'" + node.input_tensor_names[1] + "'")
        func = func + 'shape={}, '.format(gemm_weight_shape)
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        if len(node.input_tensor_names) > 2:
            z = tensor_dict[node.input_tensor_names[2]]
            oneflow_blobname_map[z] = node.input_tensor_names[2]
            # code gen for gemm bias
            gemm_bias_shape = list(tensor_dict[node.input_tensor_names[2]].shape)
            # code gen for gemm weights
            func = '{} = flow.get_variable('.format(node.input_tensor_names[2])
            func = func + 'name={}, '.format("'" + node.input_tensor_names[2] + "'")
            func = func + 'shape={}, '.format(gemm_bias_shape)
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        else:
            z = 0

        transA = False if node.attrs.get("transA", 0) == 0 else True
        transB = False if node.attrs.get("transB", 0) == 0 else True
        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)

        #code gen for gemm
        oneflow_blobname_map[x] = node.input_tensor_names[0]
        oneflow_blobname_map[y] = node.input_tensor_names[1]
        func = '{} = '.format(node.output_tensor_names[0])
        func = func + '{} * '.format(alpha)
        func = func + 'flow.linalg.matmul('
        func = func + node.input_tensor_names[0] + ', '
        func = func + node.input_tensor_names[1] + ', '
        func = func + 'transpose_a={}, '.format(transA)
        func = func + 'transpose_b={}) '.format(transB)

        if z not in oneflow_blobname_map:
            func = func + ' + {} * {}\n'.format(beta, z)
        else:
            func = func + ' + {} * {}\n'.format(beta, node.input_tensor_names[2])
        
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        return [
            alpha * flow.linalg.matmul(x, y, transpose_a=transA, transpose_b=transB)
            + beta * z
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

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("MatMul")
@flow_func(flow.linalg.matmul)
class MatMul(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        y = tensor_dict[node.input_tensor_names[1]]

        if y not in oneflow_blobname_map:
            # code gen for matmul B
            matmul_weight_shape = list(tensor_dict[node.input_tensor_names[1]].shape)
            # code gen for matmul weight_initializer
            func = 'matmul_initializer = flow.truncated_normal(0.1)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            #code gen for matmul weight_regularizer
            func = 'matmul_regularizer = flow.regularizers.l2(0.0005)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            # code gen for matmul weight_shape
            # code gen for matmul weights
            func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
            func = func + 'name={}, '.format("'" + node.input_tensor_names[1] + "'")
            func = func + 'shape={}, '.format(matmul_weight_shape)
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            
            oneflow_blobname_map[y] = node.input_tensor_names[1]

        # TODO BBuf: add broadcast code_gen
        if len(y.shape) > len(x.shape):
            broadcast_shape = y.shape[:-2] + x.shape[-2:]
            constant_for_broadcast = flow.constant(
                value=0, dtype=flow.float32, shape=broadcast_shape
            )
            func = '{}_broadcast_shape = flow.constant(value=0, dtype=flow.float32, shape={})\n'.format(node.input_tensor_names[0], broadcast_shape)
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            func = '{} = flow.math.broadcast_to_compatible_with({}, [{}_broadcast_shape])\n'.format(node.input_tensor_names[0], node.input_tensor_names[0], node.input_tensor_names[0])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            x = flow.math.broadcast_to_compatible_with(x, [constant_for_broadcast])
            if x not in oneflow_blobname_map:
                oneflow_blobname_map[x] = node.input_tensor_names[0]
        elif len(x.shape) > len(y.shape):
            broadcast_shape = x.shape[:-2] + y.shape[-2:]
            constant_for_broadcast = flow.constant(
                value=0, dtype=flow.float32, shape=broadcast_shape
            )
            func = '{}_broadcast_shape= flow.constant(value=0, dtype=flow.float32, shape={})\n'.format(node.input_tensor_names[1], broadcast_shape)
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            func = '{} = flow.math.broadcast_to_compatible_with({}, [{}_broadcast_shape])\n'.format(node.input_tensor_names[1], node.input_tensor_names[1], node.input_tensor_names[1])
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
            y = flow.math.broadcast_to_compatible_with(y, [constant_for_broadcast])
            if y not in oneflow_blobname_map:
                oneflow_blobname_map[y] = node.input_tensor_names[1]
        return cls.run_onnx_node(node, tensor_dict, inputs=(x, y), **kwargs)


@onnx_op("Clip")
class Clip(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        x_dtype = x.dtype
        if cls.SINCE_VERSION < 11:
            # min/max were required and passed as attributes
            clip_value_min = node.attrs.get("min", None)
            clip_value_max = node.attrs.get("max", None)
        else:
            # min/max are optional and passed as input_tensor_names
            init_dict = kwargs["init_dict"]
            clip_value_min = (
                init_dict[node.input_tensor_names[1]].item()
                if len(node.input_tensor_names) > 1 and node.input_tensor_names[1] != ""
                else None
            )
            clip_value_max = (
                init_dict[node.input_tensor_names[2]].item()
                if len(node.input_tensor_names) > 2 and node.input_tensor_names[2] != ""
                else None
            )

        if x not in oneflow_blobname_map:
            oneflow_blobname_map[x] = node.input_tensor_names[0]
        func = '{} = flow.math.clip_by_value({}, {}, {})\n'.format(node.output_tensor_names[0], node.input_tensor_names[0], clip_value_min, clip_value_max)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        y = flow.math.clip_by_value(x, clip_value_min, clip_value_max)

        return y

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Sqrt")
@flow_func(flow.math.sqrt)
class Sqrt(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Erf")
@flow_func(flow.math.erf)
class Erf(BackendHandler):
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)


@onnx_op("Cast")
@flow_func(flow.cast)
class Cast(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        dtype = onnx_util.Onnx2FlowDtype(node.attrs.pop("to"))
        node.attrs["dtype"] = dtype
        return cls.run_onnx_node(node, tensor_dict, **kwargs)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Abs")
class Abs(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        return flow.math.abs(x)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Exp")
class Exp(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        return flow.math.exp(x)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Reciprocal")
class Reciprocal(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        return 1.0 / x

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Floor")
class Floor(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        return flow.math.floor(x)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("ArgMax")
class ArgMax(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axis = node.attrs.get("axis")
        return flow.math.argmax(x, axis=axis)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("ArgMin")
class ArgMin(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        x = flow.math.negative(x)
        axis = node.attrs.get("axis")
        return flow.math.argmax(x, axis=axis)

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Range")
@flow_func(flow.range)
class Range(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Greater")
@flow_func(flow.math.greater)
class Greater(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Less")
@flow_func(flow.math.less)
class Less(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Softplus")
@flow_func(flow.math.softplus)
class Softplus(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
@onnx_op("Neg")
@flow_func(flow.math.negative)
class Neg(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Ceil")
@flow_func(flow.math.ceil)
class Ceil(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Where")
@flow_func(flow.where)
class Where(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Equal")
@flow_func(flow.math.equal)
class Equal(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Sign")
@flow_func(flow.math.sign)
class Sign(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("NonZero")
@flow_func(flow.nonzero)
class NonZero(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
@onnx_op("Acos")
@flow_func(flow.math.acos)
class Acos(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Acosh")
@flow_func(flow.math.acosh)
class AcosH(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Asin")
@flow_func(flow.math.asin)
class Asin(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Atan")
@flow_func(flow.math.atan)
class Atan(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Cos")
@flow_func(flow.math.cos)
class Cos(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Elu")
@flow_func(flow.nn.elu)
class Elu(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Exp")
@flow_func(flow.math.exp)
class Exp(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Log")
@flow_func(flow.math.log)
class Log(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("LogSoftmax")
@flow_func(flow.nn.logsoftmax)
class LogSoftmax(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("ReduceLogSumExp")
@flow_func(flow.math.reduce_logsumexp)
class ReduceLogSumExp(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        axis = node.attrs.pop("axes")
        node.attrs["axis"] = axis
        keepdims = bool(node.attrs.pop("keepdims"))
        node.attrs["keepdims"] = keepdims
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Round")
@flow_func(flow.math.round)
class Round(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Sin")
@flow_func(flow.math.sin)
class Sin(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("Tan")
@flow_func(flow.math.tan)
class Tan(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Tanh")
@flow_func(flow.math.tanh)
class Tanh(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return cls.run_onnx_node(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)