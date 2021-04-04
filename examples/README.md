## oneflow_onnx 使用示例

我们分别在oneflow2onnx和x2oneflow文件夹下提供了多种经典网络转换为OneFlow网络的示例，需导出ONNX或者转换自定义的TensorFlow/Pytorch/Paddle网络可以对应修改。
### oneflow2onnx

提供了AlexNet，MobileNetV2，ResNet50三个转换示例，要转换自定义的OneFlow网络可以对应修改

### x2oneflow

x2oneflow目前可以实现在OneFlow中加载TensorFlow/Pytorch/Paddle的各种模型用于训练推理

#### tensorflow2oneflow
提供了AlexNet，VGGNet，ResNet 等多种转换示例，将TensorFlow2.0的模型转换为OneFlow模型。
#### pytorch2oneflow

提供了AlexNet，VGGNet，ResNet 等多种转换示例，将Pytroch的模型转换为OneFlow模型。
#### paddle2oneflow

提供了AlexNet，VGGNet，ResNet 等多种转换示例，将PaddlePaddle的模型转换为OneFlow模型。

