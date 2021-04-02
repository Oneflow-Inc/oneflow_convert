# X2OneFlow 支持的OP列表

> 目前X2ONNX 支持60+的ONNX OP，我们在下面的列表中列出了目前OneFlow2ONNX支持的全部OP



## ONNX

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Conv      | 2 |BatchNormalization| 3  |    MaxPool    | 4 | AvgPool    |
| 5  | Concat    | 6 |   ReLU           | 7  |AdaptiveMaxPool| 8 | Softmax    |
| 9  | Unsqueeze | 10 | Transpose       | 11 | Clip          | 12 | Gather    |
| 13 | Slice     | 14 | Split           | 15 | Flatten       | 16 | Add       |
| 17 | Sub       | 18 | Mul             | 19 | Div           | 20 |Sqrt       |
| 21 |Pow        | 22 | TanH            | 23 | Sigmoid       | 24 | Cast      |
| 25 | Pad       | 26 | Reduce_Mean     | 27 | Reshape       | 28 | AdaptiveAvgPool  |
|29 | Squeeze    | 30 | Expand          | 31 | Gather        | 32 | Slice   |
|33 | Split      | 34 | Min             | 35 | Max           | 36 | Constant |
|37 | HardSigmoid| 38 | Gemm            | 39 | MatMul        | 40 | Erf      |
|41 | Cast       | 42 | GlobalMaxPool   | 43 | GlobalAveragePool |44|ReduceMax|

## TensorFlow
| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Conv      | 2 |BatchNormalization| 3  |    MaxPool    | 4 | AvgPool    |
| 5  | Concat    | 6 |   ReLU           | 7  |AdaptiveMaxPool| 8 | Softmax    |
| 9  | Unsqueeze | 10 | Transpose       | 11 | Clip          | 12 | Gather    |
| 13 | Slice     | 14 | Split           | 15 | Flatten       | 16 | Add       |
| 17 | Sub       | 18 | Mul             | 19 | Div           | 20 |Sqrt       |
| 21 |Pow        | 22 | TanH            | 23 | Sigmoid       | 24 | Cast      |
| 25 | Pad       | 26 | Reduce_Mean     | 27 | Reshape       | 28 | AdaptiveAvgPool |

- 分组卷积存在问题，已给TensorFlow2ONNX团队PR。

## Pytorch

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Conv      | 2 |BatchNormalization| 3  |    MaxPool    | 4 | AvgPool    |
| 5  | Concat    | 6 |   ReLU           | 7  |AdaptiveMaxPool| 8 | Softmax    |
| 9  | Unsqueeze | 10 | Transpose       | 11 | Clip          | 12 | Gather    |
| 13 | Slice     | 14 | Split           | 15 | Flatten       | 16 | Add       |
| 17 | Sub       | 18 | Mul             | 19 | Div           | 20 |Sqrt       |
| 21 |Pow        | 22 | TanH            | 23 | Sigmoid       | 24 | Cast      |
| 25 | Pad       | 26 | Reduce_Mean     | 27 | Reshape       | 28 | AdaptiveAvgPool |

## PaddlePaddle

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1  | Conv      | 2 |BatchNormalization| 3  |    MaxPool    | 4 | AvgPool    |
| 5  | Concat    | 6 |   ReLU           | 7  |AdaptiveMaxPool| 8 | Softmax    |
| 9  | Unsqueeze | 10 | Transpose       | 11 | Clip          | 12 | Gather    |
| 13 | Slice     | 14 | Split           | 15 | Flatten       | 16 | Add       |
| 17 | Sub       | 18 | Mul             | 19 | Div           | 20 |Sqrt       |
| 21 |Pow        | 22 | TanH            | 23 | Sigmoid       | 24 | Cast      |
| 25 | Pad       | 26 | Reduce_Mean     | 27 | Reshape       | 28 | AdaptiveAvgPool |
