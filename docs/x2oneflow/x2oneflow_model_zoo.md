# X2OneFlow模型测试库

目前X2OneFlow支持40+的TensorFlow OP，30+的Caffe Layer，40+的ONNX OP，110+的PyTorch Aten，10+的PyTorch Prim，覆盖了大部分CV分类模型常用的操作。我们在如下模型列表中测试了X2OneFlow的转换。

## Pytorch

| 模型         | 是否支持 |
| ------------ | -------- |
| AlexNet      | Yes      |
| VGGNet       | Yes      |
| GoogleNet    | Yes      |
| ResNet       | Yes      |
| ResNext      | Yes      |
| SENet        | Yes      |
| MobileNetV1  | Yes      |
| MobileNetV2  | Yes      |
| MobileNetV3  | Yes      |
| RegNet       | Yes      |
| DenseNet     | Yes      |
| EfficientNet | Yes      |
| InceptionNet | Yes      |
| ShuffleNetV1 | Yes      |
| ShuffleNetV2 | Yes      |
| SqueezeNet   | Yes      |

## TensorFlow

| 模型         | 是否支持 |
| ------------ | -------- |
| VGGNet       | Yes      |
| ResNet       | Yes      |
| ResNetV2     | Yes      |
| XceptionNet  | Yes      |
| MobileNetV1  | Yes      |
| MobileNetV2  | Yes      |
| MobileNetV3  | Yes      |
| DenseNet     | Yes      |
| EfficientNet | Yes      |
| InceptionNet | Yes      |

## PaddlePaddle

| 模型               | 是否支持                                                     |
| ------------------ | ------------------------------------------------------------ |
| AlexNet            | Yes                                                          |
| VGGNet             | Yes                                                          |
| GoogleNet          | Yes                                                          |
| ResNet             | Yes                                                          |
| ResNext            | Yes                                                          |
| SE_ResNext         | Yes                                                          |
| SENet              | Yes                                                          |
| MobileNetV1        | Yes                                                          |
| MobileNetV2        | Yes                                                          |
| MobileNetV3        | Yes                                                          |
| RegNet             | Yes                                                          |
| DenseNet           | No（msg: "op_name: Concat_58 already exist in job: job_eval"） |
| EfficientNet       | Yes                                                          |
| InceptionNet       | Yes                                                          |
| ShuffleNetV2       | Yes                                                          |
| SqueezeNet         | Yes                                                          |
| DPNNet             | Yes                                                          |
| DarkNet            | Yes                                                          |
| GhostNet           | Yes                                                          |
| RepVGG             | Yes                                                          |
| XceptionNet        | Yes                                                          |
| Xception_DeepLab   | Yes                                                          |
| Vision_Transformer | No（"op_name: Constant_20 already exist in job: job_eval"）  |
| Res2Net            | No（split op bug，working）                                  |
| Unet               | No（OneFlow的上采样OP和Paddle未对齐）                        |


- 模型的测试代码均可以在本工程的examples中找到