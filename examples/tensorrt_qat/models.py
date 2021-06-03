import oneflow as flow
import oneflow.typing as tp
import oneflow.core.operator.op_conf_pb2 as op_conf_util


def Lenet(data):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
        use_bias=False,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="VALID", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
        use_bias=False,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="VALID", name="pool2", data_format="NCHW"
    )
    # fc is replaced by conv to support tensorrt7
    hidden1 = flow.layers.conv2d(
        pool2, 512, 7, padding="VALID", name="hidden1", use_bias=False
    )
    hidden2 = flow.layers.conv2d(
        hidden1, 10, 1, padding="VALID", name="hidden2", use_bias=False
    )
    reshape = flow.reshape(hidden2, [hidden2.shape[0], -1])
    return reshape


g_trainable = False


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
):

    if data_format == "NCHW":
        weight_shape = (
            int(filters),
            int(input.shape[1] / group_num),
            int(kernel_size[0]),
            int(kernel_size[0]),
        )
    elif data_format == "NHWC":
        weight_shape = (
            int(filters),
            int(kernel_size[0]),
            int(kernel_size[0]),
            int(input.shape[3] / group_num),
        )
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input,
        weight,
        strides,
        padding,
        data_format,
        dilation_rate,
        group_num,
        name=name,
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            model_name="bias",
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output


def _batch_norm(
    inputs, axis, momentum, epsilon, center=True, scale=True, trainable=True, name=None
):

    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        trainable=trainable,
        training=trainable,
        name=name,
    )


def _relu6(data, prefix):
    return flow.clip_by_value(data, 0, 6, name="%s-relu6" % prefix)


def mobilenet_unit(
    data,
    num_filter=1,
    kernel=(1, 1),
    stride=(1, 1),
    pad=(0, 0),
    num_group=1,
    data_format="NCHW",
    if_act=True,
    use_bias=False,
    prefix="",
):
    conv = _conv2d_layer(
        name=prefix,
        input=data,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        data_format=data_format,
        dilation_rate=1,
        activation=None,
        use_bias=use_bias,
    )
    if data_format == "NCHW":
        axis = 1
    elif data_format == "NHWC":
        axis = 3
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')

    bn = _batch_norm(
        conv,
        axis=axis,
        momentum=0.97,
        epsilon=1e-3,
        name="%s-BatchNorm" % prefix,
        trainable=g_trainable,
    )
    if if_act:
        act = _relu6(bn, prefix)
        return act
    else:
        return bn


def conv(
    data,
    num_filter=1,
    kernel=(1, 1),
    stride=(1, 1),
    pad=(0, 0),
    num_group=1,
    data_format="NCHW",
    use_bias=False,
    prefix="",
):
    # return _conv2d_layer(name='%s-conv2d'%prefix, input=data, filters=num_filter, kernel_size=kernel, strides=stride, padding=pad, group_num=num_group, dilation_rate=1, activation=None, use_bias=use_bias)
    return _conv2d_layer(
        name=prefix,
        input=data,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        group_num=num_group,
        data_format=data_format,
        dilation_rate=1,
        activation=None,
        use_bias=use_bias,
    )


def shortcut(data_in, data_residual, prefix):
    out = flow.math.add(data_in, data_residual)
    return out


def inverted_residual_unit(
    data,
    num_in_filter,
    num_filter,
    ifshortcut,
    stride,
    kernel,
    pad,
    expansion_factor,
    prefix,
    data_format="NCHW",
    has_expand=1,
):
    num_expfilter = int(round(num_in_filter * expansion_factor))
    if has_expand:
        channel_expand = mobilenet_unit(
            data=data,
            num_filter=num_expfilter,
            kernel=(1, 1),
            stride=(1, 1),
            pad="valid",
            num_group=1,
            data_format=data_format,
            if_act=True,
            prefix="%s-expand" % prefix,
        )
    else:
        channel_expand = data
    bottleneck_conv = mobilenet_unit(
        data=channel_expand,
        num_filter=num_expfilter,
        stride=stride,
        kernel=kernel,
        pad=pad,
        num_group=num_expfilter,
        data_format=data_format,
        if_act=True,
        prefix="%s-depthwise" % prefix,
    )
    linear_out = mobilenet_unit(
        data=bottleneck_conv,
        num_filter=num_filter,
        kernel=(1, 1),
        stride=(1, 1),
        pad="valid",
        num_group=1,
        data_format=data_format,
        if_act=False,
        prefix="%s-project" % prefix,
    )

    if ifshortcut:
        out = shortcut(data_in=data, data_residual=linear_out, prefix=prefix,)
        return out
    else:
        return linear_out


MNETV2_CONFIGS_MAP = {
    (224, 224): {
        "firstconv_filter_num": 32,
        # t, c, s
        "bottleneck_params_list": [
            (1, 16, 1, False),
            (6, 24, 2, False),
            (6, 24, 1, True),
            (6, 32, 2, False),
            (6, 32, 1, True),
            (6, 32, 1, True),
            (6, 64, 2, False),
            (6, 64, 1, True),
            (6, 64, 1, True),
            (6, 64, 1, True),
            (6, 96, 1, False),
            (6, 96, 1, True),
            (6, 96, 1, True),
            (6, 160, 2, False),
            (6, 160, 1, True),
            (6, 160, 1, True),
            (6, 320, 1, False),
        ],
        "filter_num_before_gp": 1280,
    }
}


class MobileNetV2(object):
    def __init__(self, data_wh, multiplier, **kargs):
        super(MobileNetV2, self).__init__()
        self.data_wh = data_wh
        self.multiplier = multiplier
        if self.data_wh in MNETV2_CONFIGS_MAP:
            self.config_map = MNETV2_CONFIGS_MAP[self.data_wh]
        else:
            self.config_map = MNETV2_CONFIGS_MAP[(224, 224)]

    def build_network(
        self, input_data, data_format, class_num=1000, prefix="", **configs
    ):
        self.config_map.update(configs)

        # input_data = flow.math.multiply(input_data, 2.0 / 255.0)
        # input_data = flow.math.add(input_data, -1)

        if data_format == "NCHW":
            input_data = flow.transpose(input_data, name="transpose", perm=[0, 3, 1, 2])
        first_c = int(round(self.config_map["firstconv_filter_num"] * self.multiplier))
        first_layer = mobilenet_unit(
            data=input_data,
            num_filter=first_c,
            kernel=(3, 3),
            stride=(2, 2),
            pad="same",
            data_format=data_format,
            if_act=True,
            prefix=prefix + "-Conv",
        )

        last_bottleneck_layer = first_layer
        in_c = first_c
        for i, layer_setting in enumerate(self.config_map["bottleneck_params_list"]):
            t, c, s, sc = layer_setting
            if i == 0:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c * self.multiplier)),
                    ifshortcut=sc,
                    stride=(s, s),
                    kernel=(3, 3),
                    pad="same",
                    expansion_factor=t,
                    prefix=prefix + "-expanded_conv",
                    data_format=data_format,
                    has_expand=0,
                )
                in_c = int(round(c * self.multiplier))
            else:
                last_bottleneck_layer = inverted_residual_unit(
                    data=last_bottleneck_layer,
                    num_in_filter=in_c,
                    num_filter=int(round(c * self.multiplier)),
                    ifshortcut=sc,
                    stride=(s, s),
                    kernel=(3, 3),
                    pad="same",
                    expansion_factor=t,
                    data_format=data_format,
                    prefix=prefix + "-expanded_conv_%d" % i,
                )
                in_c = int(round(c * self.multiplier))

        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            # num_filter=int(1280 * self.multiplier) if self.multiplier > 1.0 else 1280,
            # gr to confirm
            num_filter=int(256 * self.multiplier) if self.multiplier > 1.0 else 256,
            kernel=(1, 1),
            stride=(1, 1),
            pad="valid",
            data_format=data_format,
            if_act=True,
            prefix=prefix + "-Conv_1",
        )
        base_only = True
        if base_only:
            return last_fm
        else:
            raise NotImplementedError

    def __call__(
        self, input_data, class_num=1000, prefix="", layer_out=None, **configs
    ):
        sym = self.build_network(
            input_data, class_num=class_num, prefix=prefix, **configs
        )
        if layer_out is None:
            return sym

        internals = sym.get_internals()
        if type(layer_out) is list or type(layer_out) is tuple:
            layers_out = [
                internals[layer_nm.strip() + "_output"] for layer_nm in layer_out
            ]
            return layers_out
        else:
            layer_out = internals[layer_out.strip() + "_output"]
            return layer_out


def Mobilenet(
    input_data, data_format="NCHW", num_classes=1000, multiplier=1.0, prefix=""
):
    mobilenetgen = MobileNetV2((224, 224), multiplier=multiplier)
    layer_out = mobilenetgen(
        input_data,
        data_format=data_format,
        class_num=num_classes,
        prefix=prefix + "-MobilenetV2",
        layer_out=None,
    )
    return layer_out


def get_lenet_job_function(
    func_type: str = "train", enable_qat: bool = True, batch_size: int = 100
):
    func_config = flow.FunctionConfig()
    func_config.cudnn_conv_force_fwd_algo(1)
    if enable_qat:
        func_config.enable_qat(True)
        func_config.qat.symmetric(True)
        func_config.qat.per_channel_weight_quantization(False)
        func_config.qat.moving_min_max_stop_update_after_iters(1000)
        func_config.qat.target_backend("tensorrt7")
    if func_type == "train":

        @flow.global_function(type="train", function_config=func_config)
        def train_job(
            images: tp.Numpy.Placeholder((batch_size, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((batch_size,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = Lenet(images)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
            return loss

        return train_job
    else:

        @flow.global_function(type="predict", function_config=func_config)
        def eval_job(
            images: tp.Numpy.Placeholder((batch_size, 1, 28, 28), dtype=flow.float),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = Lenet(images)

            return logits

        return eval_job


def get_mobilenet_job_function(
    func_type: str = "train", enable_qat: bool = True, batch_size: int = 100
):
    func_config = flow.FunctionConfig()
    func_config.cudnn_conv_force_fwd_algo(1)
    if enable_qat:
        func_config.enable_qat(True)
        func_config.qat.symmetric(True)
        func_config.qat.per_channel_weight_quantization(False)
        func_config.qat.moving_min_max_stop_update_after_iters(1000)
        func_config.qat.target_backend("tensorrt7")
    if func_type == "train":

        @flow.global_function(type="train", function_config=func_config)
        def train_job(
            images: tp.Numpy.Placeholder((batch_size, 1, 224, 224), dtype=flow.float),
            labels: tp.Numpy.Placeholder((batch_size,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = Mobilenet(images, num_classes=100)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
            return loss

        return train_job
    else:

        @flow.global_function(type="predict", function_config=func_config)
        def eval_job(
            images: tp.Numpy.Placeholder((batch_size, 1, 224, 224), dtype=flow.float),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = Mobilenet(images, num_classes=10)

            return logits

        return eval_job


LENET_MODEL_QAT_DIR = "./lenet_model_qat_dir"
MOBILENET_MODEL_QAT_DIR = "./mobilenet_model_qat_dir"
