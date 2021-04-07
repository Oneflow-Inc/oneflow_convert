# X2OneFlow 支持的OP列表

> 目前X2OneFlow 支持40+的ONNX OP，30+的Tensorflow/Pytorch/PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。注意我们支持的OP和模型均为动态图API下的OP和模型，要求PaddlePaddle的版本>=2.0.0，TensorFlow>=2.0.0，Pytorch无明确版本要求。

**注：** 目前，部分OP暂未支持，如您在转换过程中出现OP不支持的情况，可自行添加或反馈给我们。欢迎通过[ISSUE](https://github.com/Oneflow-Inc/oneflow_convert_tools/issues/new)反馈的方式告知我们(模型名，代码实现或模型获取方式)，我们会及时跟进：）



## ONNX

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Conv      | 2 |BatchNormalization| 3  |    MaxPool    | 4 | AveragePool|
| 5  | Concat    | 6 |   ReLU           | 7  |AdaptiveMaxPool| 8 | Softmax    |
| 9  | Unsqueeze | 10 | Transpose       | 11 | Clip          | 12 | Gather    |
| 13 | Slice     | 14 | Split           | 15 | Flatten       | 16 | Add       |
| 17 | Sub       | 18 | Mul             | 19 | Div           | 20 |Sqrt       |
| 21 |Pow        | 22 | Tanh            | 23 | Sigmoid       | 24 | Cast      |
| 25 | Pad       | 26 | ReduceMean     | 27 | Reshape        | 28 | AdaptiveAvgPool  |
|29 | Squeeze    | 30 | Expand          | 31 | Gather        | 32 | Slice   |
|33 | Split      | 34 | Min             | 35 | Max           | 36 | Constant |
|37 | HardSigmoid| 38 | Gemm            | 39 | MatMul        | 40 | Erf      |
|41 | ~~Cast~~   | 42 | GlobalMaxPool   | 43 | GlobalAveragePool |44|ReduceMax|
|45 | Identity   | 46 | Rsqrt           | 47 | LeakyRelu     | 48 | Abs       |
|49 | Exp        | 50 | Reciprocal      | 51 | Floor         | 52 | ArgMax    |
|53 | Range      | 54 | Greator         | 55 | Less          | 56 | Softplus  |
|57 | Neg        | 58 | Ceil            | 59 | Where         | 60 | Equal     |

## TensorFlow


| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Relu             | 2  | Relu6          | 3  | Shape          | 4  | Abs                   |
| 5  | Sigmoid          | 6  | Exp            | 7  | Rsqrt          | 8  | Swish                 |
| 9  | Tanh             | 10 | LeakyRelu      | 11 | Add            | 12 | Greater               |
| 13 | Sub              | 14 | Maximum        | 15 | Mul            | 16 | FloorDiv              |
| 17 | Pow              | 18 | Const          | 19 | Transpose      | 20 | BatchNormalization    |
| 21 | Conv2D           | 22 | BiasAdd        | 23 | MaxPool        | 24 | DepthwiseConv2D       |
| 25 | Reshape          | 26 | AvgPool        | 27 | Where          | 28 | SquaredDifference     |
| 29 | Neg              | 30 | Ceil           | 31 | Pad            | 32 | ~~ResizeBilinear~~    |
| 33 | ReduceMean       | 34 | MatMul         | 35 | ArgMax         | 36 | ExpandDims            |
| 37 | Slice            | 38 | Sum            | 39 | Max            | 40 | ~~LessEqual~~         |
| 41 | ~~Cast~~         | 42 | Split          | 43 | Squeeze        | 44 | ~~ResizeNearestNeighbor~~ |
| 45 | Softmax          | 46 | Range          | 47 | Size           | 48 |  Sqrt                   |
| 49 | Identity         | 50 |~~GreaterEqual~~| 51 | Equal          | 52 | Minimum               |
| 53 |                  | 54 | Fill           | 55 | Floor          | 56 |                       |
| 57 | Sqrt             | 58 | Softplus       | 59 | Erf            | 60 |                       |



## Pytorch

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | relu      | 2 |cat | 3  |   unsqueeze   | 4 | transpose|
| 5  | batchnorm | 6 |slice       | 7  |   gather        | 8 | clamp|
| 9  | conv2d    | 10| depthwiseconv2d| 11| flatten      | 12| add      |
| 13 | sub       | 14| mul        | 15 | div             | 16| pow      |
| 17 | sqrt      | 18| tanh       | 19 | sigmoid         | 20| erf      |
| 21 | cast      | 22| pad        | 23 | maxpool         | 24| avgpool  |
| 25 | globalavgpool| 26| globalmaxpool | 27 | reduce_mean| 28| reshape |
| 29 | softmax   |30 | relu6      | 31 | CrossEntropyLoss |


## PaddlePaddle

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | relu      | 2 |concatenate | 3  |   expand_dims   | 4 | transpose|
| 5  | batchnorm | 6 |slice       | 7  |   gather        | 8 | clip_by_value|
| 9  | conv2d    | 10| depthwiseconv2d| 11| flatten      | 12| add      |
| 13 | sub       | 14| mul        | 15 | div             | 16| pow      |
| 17 | sqrt      | 18| tanh       | 19 | sigmoid         | 20| ~~erf~~      |
| 21 | cast      | 22| pad        | 23 | maxpool         | 24| avgpool  |
| 25 | adaptiveavgpool| 26| ~~adptivemaxpool~~ | 27 | reduce_mean| 28| reshape |
| 29 | softmax   |30 | relu6      | 

相关issue：

- https://github.com/PaddlePaddle/Paddle2ONNX/issues/221
- https://github.com/PaddlePaddle/Paddle2ONNX/issues/220
