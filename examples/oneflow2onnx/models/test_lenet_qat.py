import tempfile
import numpy as np
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
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
        use_bias=False,
    )
    return flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="dense2", use_bias=False
    )


def get_job_function(
    func_type: str = "train", enable_qat: bool = True, batch_size: int = 100
):
    func_config = flow.FunctionConfig()
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


if __name__ == "__main__":
    batch_size = 100
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        batch_size, batch_size
    )
    # train
    train_job = get_job_function("train", batch_size=batch_size)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
    flow_weight_dir = tempfile.TemporaryDirectory()
    flow.checkpoint.save(flow_weight_dir.name)
    flow.clear_default_session()
    # predict   
    print(flow_weight_dir.name)
    eval_job = get_job_function("predict", batch_size=batch_size)         
    flow.load_variables(flow.checkpoint.get(flow_weight_dir.name))
    for k,v in flow.get_all_variables().items():
        print(k)
        print(v.numpy())
    data = np.ones((batch_size, 1, 28, 28))
    print(eval_job(data))
