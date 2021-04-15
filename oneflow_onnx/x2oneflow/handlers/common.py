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
import copy
import oneflow
import numpy as np

from oneflow_onnx.x2oneflow.handler import oneflow_code_gen, oneflow_blobname_map

class BroadcastMixin(object):
    @classmethod
    def explicit_broadcast(cls, inputs, axis=None, tensor_dict=None):
        x = tensor_dict[inputs[0]]
        y = tensor_dict[inputs[1]]

        if np.prod(y.shape) == 1:
            return y

        if axis is None:
            return y

        total_num_dim = len(x.get_shape())
        if axis < 0:
            axis += total_num_dim

        if axis + len(y.get_shape()) == total_num_dim:
            return y

        dims = [axis + i for i in range(len(y.get_shape()))]
        new_y = y
        for i in range(total_num_dim):
            if i not in dims:
                raise NotImplementedError()
                # new_y = tf.expand_dims(new_y, i)
        return new_y

    @classmethod
    def limited_broadcast(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.inputs[0]]
        y = tensor_dict[node.inputs[1]]
        if node.attrs.get("broadcast") == 1:
            y = cls.explicit_broadcast([x, y], node.attrs.get("axis", None))
            return [cls.run_onnx_node(node, inputs=[x, y], **kwargs)]
        return [cls.run_onnx_node(node, **kwargs)]


class ConvMixin(BroadcastMixin):
    @classmethod
    def conv(cls, node, input_dict, transpose=False):
        """ Convolution method for both conv and transposed conv
        For transposed conv,
        Attr pads is not used for input, but declares how much output is padded.
        Here, output means output from transposed conv which already pad output_padding if set.
        So the pseudo explanation for output should be:
            output = conv_transpose_output + output_padding - pads
        And conv_transpose_output shape should be:
            conv_transpose_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
        """
        x = input_dict[node.input_tensor_names[0]]
        x_shape = list(x.shape)
        x_rank = len(x_shape)
        spatial_size = x_rank - 2

        in_weights = input_dict[node.input_tensor_names[1]]
        in_weights_shape = list(in_weights.shape)
        
        # code gen for conv weight_initializer
        func = 'weight_initializer = flow.truncated_normal(0.1)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)
        #code gen for conv weight_regularizer
        func = 'weight_regularizer = flow.regularizers.l2(0.0005)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        weights_rank = len(in_weights_shape)
        if transpose:
            # Translate weights from (C x M x KH x KW) to (KH x KW X M X C)
            perm = list(range(2, weights_rank)) + [1, 0]
        else:
            # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
            perm = list(range(2, weights_rank)) + [1, 0]

        if "kernel_shape" in node.attrs.keys():
            kernel_shape = node.attrs["kernel_shape"]
            assert in_weights_shape[2:] == kernel_shape, (
                "kernel_shape "
                "attr of convolution does not match the actual weight "
                "passed to this operation, attr {}, actual {}"
            ).format(kernel_shape, in_weights_shape)
        else:
            kernel_shape = in_weights_shape[2:]

        weights = in_weights
        # code gen for conv weights
        func = '{} = flow.get_variable('.format(node.input_tensor_names[1])
        func = func + 'name={}, '.format("'"+node.input_tensor_names[1]+"'")
        func = func + 'shape={}, '.format(in_weights_shape)
        func = func + 'initializer=weight_initializer, '
        func = func + 'regularizer=weight_regularizer)\n'
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        dilations = node.attrs.get("dilations", [1] * spatial_size)
        strides = node.attrs.get("strides", [1] * spatial_size)

        pads = node.attrs.get("pads", [0, 0] * spatial_size)
        pad_flag = 0
        # Check auto_pad nonexistent or NOTSET first
        if "auto_pad" not in node.attrs or node.attrs["auto_pad"] == "NOTSET":
            if not transpose:
                if pads != [0, 0] * spatial_size:
                    x = PadMixin.get_padding_as_op(x, pads)

                    num_dim = int(len(pads) / 2)
                    flow_pads = (
                        np.transpose(np.array(pads).reshape([2, num_dim])).astype(np.int32).tolist()
                    )
                    # flow_pads = [0, 0, 0, 0] + flow_pads.flatten().tolist()
                    flow_pads = [(0, 0), (0, 0)] + flow_pads
                    func = '{}_tmp_conv_pad = flow.pad({}, paddings={})\n'.format(node.input_tensor_names[0], node.input_tensor_names[0], flow_pads)
                    pad_flag = 1
                    if func not in oneflow_code_gen:
                        oneflow_code_gen.append(func)
                pad_mode = "VALID"
            else:
                pad_mode = "NOTSET"
        # Then we use auto_pad to setup pad_mode
        elif node.attrs["auto_pad"] == "SAME_UPPER":
            pad_mode = "SAME"
        elif node.attrs["auto_pad"] == "VALID":
            pad_mode = "VALID"
        elif node.attrs["auto_pad"] == "SAME_LOWER":
            pad_mode = "SAME_LOWER"
        else:
            raise ValueError(
                "Invalid auto_pad attribute: {}".format(node.attrs["auto_pad"])
            )

        group = node.attrs.get("group", 1)

        conv = oneflow.nn.conv2d(
            x,
            weights,
            padding=pad_mode,
            strides=strides,
            dilations=dilations,
            data_format="NCHW",
            groups=group,
        )

        # code gen for conv
        oneflow_blobname_map[x] = node.input_tensor_names[0]
        oneflow_blobname_map[weights] = node.input_tensor_names[1]
        oneflow_blobname_map[conv] = node.output_tensor_names[0]
        
        func = '{} = '.format(node.output_tensor_names[0])
        func = func + 'flow.nn.conv2d('
        if pad_flag == 0:
            func = func + node.input_tensor_names[0] + ', '
        else:
            func = func + '{}_tmp_conv_pad'.format(node.input_tensor_names[0]) + ', '
        func = func + node.input_tensor_names[1] + ', '
        func = func + 'padding={}, '.format("'"+pad_mode+"'")
        func = func + 'strides={}, '.format(strides)
        func = func + 'data_format={}, '.format("'NCHW'")
        func = func + 'groups={})\n'.format(group)
        if func not in oneflow_code_gen:
            oneflow_code_gen.append(func)

        if len(node.input_tensor_names) == 2:
            output = conv
        else:
            bias = input_dict[node.input_tensor_names[2]]
            output = oneflow.nn.bias_add(conv, bias)
            oneflow_blobname_map[output] = node.output_tensor_names[0]
            # code gen for bias_add
            func = '{} = flow.get_variable('.format(node.input_tensor_names[2])
            func = func + 'name={}, '.format("'"+node.input_tensor_names[2]+"'")
            func = func + 'shape={}, '.format(list(bias.shape))
            func = func + 'initializer=weight_initializer, '
            func = func + 'regularizer=weight_regularizer)\n'
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)

            oneflow_blobname_map[bias] = node.input_tensor_names[2]
            func = '{} = '.format(node.output_tensor_names[0])
            func = func + 'flow.nn.bias_add('
            func = func + node.output_tensor_names[0] + ', '
            func = func + node.input_tensor_names[2] + ', '
            func = func + 'data_format={})\n'.format("'NCHW'")
            if func not in oneflow_code_gen:
                oneflow_code_gen.append(func)
        return [output]


class PadMixin(object):
    @classmethod
    def get_padding_as_op(cls, x, pads):
        num_dim = int(len(pads) / 2)

        flow_pads = (
            np.transpose(np.array(pads).reshape([2, num_dim])).astype(np.int32).tolist()
        )
        # flow_pads = [0, 0, 0, 0] + flow_pads.flatten().tolist()
        flow_pads = [(0, 0), (0, 0)] + flow_pads

        return oneflow.pad(x, flow_pads)


class BasicMathMixin(BroadcastMixin):
    pass


class ArithmeticMixin(BroadcastMixin):
    pass


class ReductionMixin(BroadcastMixin):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        attrs = copy.deepcopy(node.attrs)
        axis = attrs.pop("axes", None)
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            axis = axis[0]
        attrs["axis"] = axis
        # https://github.com/onnx/onnx/issues/585
        attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
        return cls.run_onnx_node(node, tensor_dict, attrs=attrs, **kwargs)
