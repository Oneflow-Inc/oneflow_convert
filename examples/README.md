## oneflow_onnx 使用示例

目前OneFlow2ONNX 支持80+的OneFlow OP导出为ONNX OP。X2OneFlow支持79个ONNX OP，50+个TensorFlow OP，80+个Pytorch OP，50+个PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。注意我们支持的OP和模型均为动态图API下的OP和模型，要求PaddlePaddle的版本>=2.0.0，TensorFlow>=2.0.0，Pytorch无明确版本要求。目前X2OneFlow已经成功转换了50+个TensorFlow/Pytorch/PaddlePaddle官方模型.我们分别在`oneflow2onnx`和`x2oneflow`文件夹下提供了所有的测试示例，需导出ONNX或者转换自定义的TensorFlow/Pytorch/PaddlePaddle网络可以对应修改使用。


