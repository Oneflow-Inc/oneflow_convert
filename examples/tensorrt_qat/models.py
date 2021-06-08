import oneflow as flow
import oneflow.typing as tp


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


def _get_regularizer(model_name):
    # all decay
    return flow.regularizers.l2(0.00004)


def _get_initializer(model_name):
    if model_name == "weight":
        return flow.variance_scaling_initializer(
            2.0, mode="fan_out", distribution="random_normal", data_format="NCHW"
        )
    elif model_name == "bias":
        return flow.zeros_initializer()
    elif model_name == "gamma":
        return flow.ones_initializer()
    elif model_name == "beta":
        return flow.zeros_initializer()
    elif model_name == "dense_weight":
        return flow.random_normal_initializer(0, 0.01)


def _batch_norm(
    inputs,
    axis,
    momentum,
    epsilon,
    center=True,
    scale=True,
    trainable=True,
    training=True,
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=_get_initializer("beta"),
        gamma_initializer=_get_initializer("gamma"),
        beta_regularizer=_get_regularizer("beta"),
        gamma_regularizer=_get_regularizer("gamma"),
        trainable=trainable,
        training=training,
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
    trainable=True,
    training=True,
    prefix="",
):
    conv = flow.layers.conv2d(
        inputs=data,
        filters=num_filter,
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilation_rate=1,
        groups=num_group,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=_get_initializer("weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("weight"),
        bias_regularizer=_get_regularizer("bias"),
        name=prefix,
    )
    bn = _batch_norm(
        conv,
        axis=1,
        momentum=0.9,
        epsilon=1e-5,
        trainable=trainable,
        training=training,
        name="%s-BatchNorm" % prefix,
    )
    if if_act:
        act = _relu6(bn, prefix)
        return act
    else:
        return bn


def shortcut(data_in, data_residual, prefix):
    out = flow.math.add(data_in, data_residual, f"{prefix}-add")
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
    trainable=True,
    training=True,
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
            trainable=trainable,
            training=training,
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
        trainable=trainable,
        training=training,
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
        trainable=trainable,
        training=training,
        prefix="%s-project" % prefix,
    )

    if ifshortcut:
        out = shortcut(data_in=data, data_residual=linear_out, prefix=prefix)
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
    def __init__(self, data_wh, multiplier, trainable=True, training=True, **kargs):
        super(MobileNetV2, self).__init__()
        self.data_wh = data_wh
        self.multiplier = multiplier
        self.trainable = trainable
        self.training = training
        if self.data_wh in MNETV2_CONFIGS_MAP:
            self.config_map = MNETV2_CONFIGS_MAP[self.data_wh]
        else:
            self.config_map = MNETV2_CONFIGS_MAP[(224, 224)]

    def build_network(
        self, input_data, data_format, class_num=1000, prefix="", **configs
    ):
        self.config_map.update(configs)

        first_c = int(round(self.config_map["firstconv_filter_num"] * self.multiplier))
        first_layer = mobilenet_unit(
            data=input_data,
            num_filter=first_c,
            kernel=(3, 3),
            stride=(2, 2),
            pad="same",
            data_format=data_format,
            if_act=True,
            trainable=self.trainable,
            training=self.training,
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
                    trainable=self.trainable,
                    training=self.training,
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
                    prefix=prefix + "-expanded_conv_%d" % i,
                    trainable=self.trainable,
                    training=self.training,
                    data_format=data_format,
                )
                in_c = int(round(c * self.multiplier))
        last_fm = mobilenet_unit(
            data=last_bottleneck_layer,
            num_filter=int(1280 * self.multiplier) if self.multiplier > 1.0 else 1280,
            kernel=(1, 1),
            stride=(1, 1),
            pad="valid",
            data_format=data_format,
            if_act=True,
            trainable=self.trainable,
            training=self.training,
            prefix=prefix + "-Conv_1",
        )
        # global average pooling
        pool_size = int(self.data_wh[0] / 32)
        pool = flow.nn.avg_pool2d(
            last_fm,
            ksize=pool_size,
            strides=1,
            padding="VALID",
            data_format="NCHW",
            name="pool5",
        )
        fc = flow.layers.dense(
            flow.reshape(pool, (pool.shape[0], -1)),
            units=class_num,
            use_bias=False,
            kernel_initializer=_get_initializer("dense_weight"),
            bias_initializer=_get_initializer("bias"),
            kernel_regularizer=_get_regularizer("dense_weight"),
            bias_regularizer=_get_regularizer("bias"),
            trainable=self.trainable,
            name=prefix + "-fc",
        )
        return fc

    def __call__(self, input_data, class_num=1000, prefix="", **configs):
        sym = self.build_network(
            input_data, class_num=class_num, prefix=prefix, **configs
        )
        return sym


def Mobilenet(
    input_data,
    channel_last=False,
    trainable=True,
    training=True,
    num_classes=1000,
    multiplier=1.0,
    prefix="",
):
    assert (
        channel_last == False
    ), "Mobilenet does not support channel_last mode, set channel_last=False will be right!"
    data_format = "NCHW"
    mobilenetgen = MobileNetV2(
        (224, 224), multiplier=multiplier, trainable=trainable, training=training
    )
    out = mobilenetgen(
        input_data, data_format=data_format, class_num=num_classes, prefix="MobilenetV2"
    )
    return out


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
                logits = Mobilenet(images, num_classes=10)
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
                logits = Mobilenet(
                    images, trainable=False, training=False, num_classes=10
                )

            return logits

        return eval_job


LENET_MODEL_QAT_DIR = "./lenet_model_qat_dir"
MOBILENET_MODEL_QAT_DIR = "./mobilenet_model_qat_dir"
