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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging
import sys

import numpy as np
from onnx import numpy_helper
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

import oneflow
import oneflow_onnx
from oneflow_onnx import constants, util
from oneflow_onnx.oneflow2onnx.graph_builder import GraphBuilder
from oneflow_onnx.oneflow2onnx.handler import flow_op
from oneflow_onnx.oneflow2onnx.handlers import nn, math

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


def _ConvertShapeNodeToInt64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    name = node.input_tensor_names[input_number]

    cast_node = ctx.InsertNewNodeOnInput(node, "Cast", name)
    cast_node.attrs["to"] = onnx_pb.TensorProto.INT64
    ctx.set_dtype(cast_node.output_tensor_names[0], onnx_pb.TensorProto.INT64)
    ctx.CopyShape(name, cast_node.output_tensor_names[0])


def _WrapConcatWithCast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output_tensor_names[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output_tensor_names[0]
        # cast each inputs to float
        for i, inp in enumerate(node.input_nodes):
            input_cast = ctx.InsertNewNodeOnInput(node, "Cast", node.input_tensor_names[i])
            input_cast.attrs["to"] = onnx_pb.TensorProto.FLOAT
            ctx.set_dtype(input_cast.output_tensor_names[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.FindOutputConsumers(node.output_tensor_names[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].op_type != "Cast":
            op_name = oneflow._oneflow_internal.UniqueStr(node.name)
            output_cast = ctx.InsertNewNodeOnOutput("Cast", output_name, name=op_name)
            output_cast.attrs["to"] = dtype
            ctx.set_dtype(output_cast.output_tensor_names[0], dtype)
            ctx.CopyShape(output_name, output_cast.output_tensor_names[0])


@flow_op("reshape", "Reshape")
class Reshape:
    @classmethod
    def Version_5(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.output_tensor_names[0])
        need_casting = dtype in [
            onnx_pb.TensorProto.INT32,
            onnx_pb.TensorProto.INT16,
            onnx_pb.TensorProto.INT64,
        ]

        shape_node = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("shape"), np.array(node.attrs.get("shape"), None))
        if node.attrs.get("shape") == []:
            shape_node = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("shape"), np.array([]).astype(np.int64))

        node.input_tensor_names = node.input_tensor_names + [shape_node.name]
        if ctx.opset >= 8 or not need_casting:
            # onnx reshape can handle the type - done
            return

        # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
        input_cast = ctx.InsertNewNodeOnInput(node, "Cast", node.input_tensor_names[0])
        input_cast.attrs["to"] = onnx_pb.TensorProto.FLOAT
        ctx.CopyShape(node.output_tensor_names[0], input_cast.output_tensor_names[0])

        # if the next node is already a cast we don't need to insert another one
        next_nodes = ctx.FindOutputConsumers(node.output_tensor_names[0])
        if len(next_nodes) != 1 or next_nodes[0].op_type != "Cast":
            op_name = oneflow._oneflow_internal.UniqueStr(node.name)
            output_cast = ctx.InsertNewNodeOnOutput("Cast", node.output_tensor_names[0], name=op_name)
            output_cast.attrs["to"] = dtype
            ctx.set_dtype(output_cast.output_tensor_names[0], dtype)
            ctx.CopyShape(node.output_tensor_names[0], output_cast.output_tensor_names[0])


@flow_op("flatten")
class Flatten:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        shape = ctx.get_shape(node.input_tensor_names[0])
        dim = len(shape)
        start_dim = node.attrs.get("start_dim", 1)
        end_dim = node.attrs.get("end_dim", -1)
        if end_dim < 0:
            end_dim += dim
        if start_dim == 1 and end_dim == dim - 1:
            ctx.RemoveNode(node.name)
            ctx.MakeNode("Flatten", [node.input_tensor_names[0]], attr={"aixs": start_dim}, outputs=[node.output_tensor_names[0]], op_name_scope=node.name, name="new_flatten")
            return
        if start_dim == 0 and end_dim == dim - 2:
            ctx.RemoveNode(node.name)
            ctx.MakeNode("Flatten", [node.input_tensor_names[0]], attr={"aixs": end_dim + 1}, outputs=[node.output_tensor_names[0]], op_name_scope=node.name, name="new_flatten")
            return

        if start_dim > 1:
            flatten_node = ctx.MakeNode("Flatten", [node.input_tensor_names[0]], attr={"aixs": 0}, op_name_scope=node.name, name="new_flatten")
            new_shape = []
            for i in range(start_dim):
                new_shape.append(shape[i])
            shape2 = 1
            for i in range(start_dim, end_dim + 1):
                shape2 *= shape[i]
            new_shape.append(shape2)
            for i in range(end_dim + 1, dim):
                new_shape.append(shape[i])
            ctx.RemoveNode(node.name)
            new_shape_name = oneflow._oneflow_internal.UniqueStr("new_shape")
            ctx.MakeConst(new_shape_name, np.array(new_shape, dtype=np.int64))
            ctx.MakeNode("Reshape", [flatten_node.output_tensor_names[0], new_shape_name], outputs=[node.output_tensor_names[0]], op_name_scope=node.name, name="new_reshape")


@flow_op("squeeze", "Squeeze")
class Squeeze:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T output = Squeeze(T input, @list(int) squeeze_dims)
        # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
        axis = node.attrs.get("axes", None)

        neg_axis = any([val < 0 for val in axis])
        if neg_axis:
            shape = ctx.get_shape(node.input_tensor_names[0])
            util.MakeSure(shape is not None, "squeeze input shape cannot be None")
            shape_len = len(shape)
            axis = [a + shape_len if a < 0 else a for a in axis]
        node.attrs["axes"] = axis

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.Version_1(ctx, node, **kwargs)


@flow_op("expand_dims", "Unsqueeze")
class ExpandDimsOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        axis = node.attrs.get("axis", None)
        node.attrs["axes"] = [axis]

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        cls.Version_1(ctx, node, **kwargs)

    @classmethod
    def Version_13(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        axis = node.attrs.get("axis", None)
        axis_node = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("axis"), np.array(axis))
        node.input_tensor_names.append(axis_node.output_tensor_names[0])


@flow_op("expand", "Expand")
class ExpandOp:
    @classmethod
    def Version_8(cls, ctx, node, **kwargs):
        shape = node.attrs.get("expand_shape")
        shape_node = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("shape"), np.array(shape).astype(np.int64))
        node.input_tensor_names.append(shape_node.output_tensor_names[0])

    @classmethod
    def Version_13(cls, ctx, node, **kwargs):
        cls.Version_8(ctx, node, **kwargs)


@flow_op("transpose", onnx_op="Transpose")
class Transpose:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input_tensor_names) > 1:
            perm = node.input_nodes[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
                ctx.RemoveInput(node, node.input_tensor_names[1])
                node.attrs["perm"] = dims
            else:
                util.MakeSure(False, "perm can't be dynamic in ONNX")
        else:
            # graph rewrite moved perm to attribute
            pass


@flow_op("concat", "Concat")
class Concat:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # old concat op has axis as input[0]
        axis_val = node.attrs.get("axis", None)

        if axis_val < 0:
            input_shape = ctx.get_shape(node.input_tensor_names[0])
            axis_val = len(input_shape) + axis_val
        node.attrs["axis"] = axis_val

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _WrapConcatWithCast(ctx, node)
            return

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.Version_1(ctx, node, **kwargs)


@flow_op("slice", "Slice")
class Slice:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        starts = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("start"), np.array(node.attrs["start"]).astype(np.int64),)
        node.input_tensor_names.append(starts.output_tensor_names[0])
        ends = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("stop"), np.array(node.attrs["stop"]).astype(np.int64),)
        node.input_tensor_names.append(ends.output_tensor_names[0])
        slice_axes = []
        input_shape = ctx.get_shape(node.input_tensor_names[0])
        for i in range(len(input_shape)):
            slice_axes.append(i)

        axes = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("axes"), np.array(slice_axes).astype(np.int64),)
        node.input_tensor_names.append(axes.output_tensor_names[0])
        steps = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("steps"), np.array(node.attrs["step"]).astype(np.int64),)
        node.input_tensor_names.append(steps.output_tensor_names[0])

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        cls.Version_1(ctx, node, **kwargs)


@flow_op("narrow", "Slice")
class Narrow:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        dim = node.attrs.get("dim", None)
        start = node.attrs.get("start", None)
        length = node.attrs.get("length", None)
        end = start + length
        slice_axes = []
        slice_starts = []
        slice_ends = []
        slice_steps = []
        input_shape = ctx.get_shape(node.input_tensor_names[0])
        for i in range(len(input_shape)):
            slice_axes.append(i)
            slice_steps.append(1)
            if i == dim:
                slice_starts.append(start)
                slice_ends.append(end)
            else:
                slice_starts.append(0)
                slice_ends.append(input_shape[i])

        starts = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("narrow_start"), np.array(slice_starts).astype(np.int64),)
        node.input_tensor_names.append(starts.output_tensor_names[0])
        ends = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("narrow_length"), np.array(slice_ends).astype(np.int64),)
        node.input_tensor_names.append(ends.output_tensor_names[0])
        axes = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("narrow_axes"), np.array(slice_axes).astype(np.int64),)
        node.input_tensor_names.append(axes.output_tensor_names[0])
        steps = ctx.MakeConst(oneflow._oneflow_internal.UniqueStr("narrow_steps"), np.array(slice_steps).astype(np.int64),)
        node.input_tensor_names.append(steps.output_tensor_names[0])

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        cls.Version_1(ctx, node, **kwargs)


@flow_op("gather_nd", onnx_op="GatherND", flow_ibns=["params", "indices"])
class GatherND:
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # indicies input
        input1 = node.input_tensor_names[1]
        target_dtype = TensorProto.INT64
        if ctx.get_dtype(input1) != TensorProto.INT64:
            inp_cast = ctx.InsertNewNodeOnInput(node, "Cast", input1, to=target_dtype)
            ctx.CopyShape(input1, inp_cast.output_tensor_names[0])
            ctx.set_dtype(inp_cast.output_tensor_names[0], target_dtype)


@flow_op("cast", "Cast")
class Cast:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        dst = node.attrs.get("dtype", None)
        node.attrs["to"] = dst

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        cls.Version_6(ctx, node, **kwargs)


@flow_op(["identity", "amp_white_identity", "amp_black_identity"], "Identity")
class Identity:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass


@flow_op("constant", "Constant")
class Constant:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        floating_value = node.attrs.get("floating_value", 0.0)
        integer_value = node.attrs.get("integer_value", 0)
        is_floating_value = node.attrs.get("is_floating_value", False)
        shape = node.attrs.get("shape", None)
        if is_floating_value:
            values = np.full(shape=shape, fill_value=floating_value, dtype=np.float32)
        else:
            values = np.full(shape=shape, fill_value=integer_value, dtype=np.int64)
        output_name = node.output_tensor_names[0]
        ctx.RemoveNode(node.name)
        if is_floating_value:
            ctx.MakeConst(output_name, values)
        else:
            ctx.MakeConst(output_name, values)


@flow_op("gather", "Gather", flow_ibns=["in", "indices"])
class Gather:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.input_tensor_names[1])
        assert dtype == onnx_pb.TensorProto.INT32 or dtype == onnx_pb.TensorProto.INT64, "onnx gather only support int32/int64 indices."

    @classmethod
    def Version_13(cls, ctx, node, **kwargs):
        cls.Version_1(ctx, node, **kwargs)


@flow_op("where", "Where", flow_ibns=["condition", "x", "y"])
class Where:
    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def Version_16(cls, ctx, node, **kwargs):
        pass
