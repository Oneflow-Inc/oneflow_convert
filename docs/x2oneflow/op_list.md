# X2OneFlow 支持的OP列表

> 目前X2OneFlow 支持40+的ONNX OP，30+的Tensorflow/Pytorch/PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。OP的单元测试代码会逐渐移步到工程的examples目录下，并支持更多的OP。



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
|41 | Cast       | 42 | GlobalMaxPool   | 43 | GlobalAveragePool |44|ReduceMax|
|45 | Identity   |
## TensorFlow

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | relu      | 2 |concatenate | 3  |   expand_dims   | 4 | transpose|
| 5  | batchnorm | 6 |slice       | 7  |   gather        | 8 | clip_by_value|
| 9  | conv2d    | 10| depthwiseconv2d| 11| flatten      | 12| add      |
| 13 | sub       | 14| mul        | 15 | div             | 16| pow      |
| 17 | sqrt      | 18| tanh       | 19 | sigmoid         | 20| erf      |
| 21 | cast      | 22| pad        | 23 | maxpool         | 24| avgpool  |
| 25 | globalavgpool| 26| globalmaxpool | 27 | reduce_mean| 28| reshape |
| 29 | softmax   |30 | relu6      |   

- 分组卷积存在问题，已给TensorFlow2ONNX团队PR。

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
