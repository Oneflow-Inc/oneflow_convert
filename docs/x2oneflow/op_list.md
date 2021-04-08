# X2OneFlow 支持的OP列表

> 目前X2OneFlow 支持79个ONNX OP，50+个TensorFlow OP，80+个Pytorch OP，50+个PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。注意我们支持的OP和模型均为动态图API下的OP和模型，要求PaddlePaddle的版本>=2.0.0，TensorFlow>=2.0.0，Pytorch无明确版本要求。

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
| 25 | Pad       | 26 | ReduceMean      | 27 | Reshape       | 28 | AdaptiveAvgPool|
|29 | Squeeze    | 30 | Expand          | 31 | Gather        | 32 | Slice   |
|33 | Split      | 34 | Min             | 35 | Max           | 36 | Constant |
|37 | HardSigmoid| 38 | Gemm            | 39 | MatMul        | 40 | Erf      |
|41 | ~~Cast~~   | 42 | GlobalMaxPool   | 43 | GlobalAveragePool |44|ReduceMax|
|45 | Identity   | 46 | Rsqrt           | 47 | LeakyRelu     | 48 | Abs       |
|49 | Exp        | 50 | Reciprocal      | 51 | Floor         | 52 | ArgMax    |
|53 | Range      | 54 | Greator         | 55 | Less          | 56 | Softplus  |
|57 | Neg        | 58 | Ceil            | 59 | Where         | 60 | Equal     |
|61 | Sign       | 62 | NonZero         | 63 | Acos          | 64 | Acosh     |
|65 | ArgMin     | 66 | Asin            | 67 | Atan          | 68 | Cos       |
|69 | Elu        | 70 | Exp             | 71 | Log           | 72 | LogSoftmax|
|73 |ReduceLogSumExp|74| ReduceMin      | 75 | ReduceProd    | 76 | Round     |
|77 | Sin        | 78 | Tanh            | 79 |Tan            |


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
| 1 | BatchNorm | 2 |  ConstantPad2d       | 3 | Conv2D              | 4 | Dropout   |
| 5 | MaxPool2d | 6 |  adaptive_avg_pool2d | 7 | adaptive_max_pool2d | 8 | AvgPool2d |
| 9 | abs       | 10|  absolute            | 11| acos                | 12 | add      |
| 13| addmm     | 14|  arange              | 15| argmax              | 16 | argmin   |
| 17| asin      | 18|  atan                | 19| baddbmm             | 20 | cat      |
| 21| ceil      | 22|  ~~celu~~            | 23| clamp               | 24 | clamp_max|
| 25| clamp_min | 26| concat               | 27| cos                 | 28 | ~~cumsum~~|
| 29| div       | 30| elu                  | 31| eq                  | 32 | erf      |
| 33| exp       | 34| ~~expand~~           | 35| flatten             | 36 | floor    |
| 37|floor_divide|38| full                 | 39| full_like           | 40 | gather   |
| 41| ~~ge~~    | 42| gelu                 | 43| ~~GroupNorm~~       | 44 |~~hardswish~~|
| 45| hardtanh  | 46| ~~instance_norm~~    | 47| ~~interpolate~~     | 48 | ~~layer_norm~~|
| 49| leaky_relu| 50| log                  | 51| log1p               | 52 | log2     |
| 53| log_softmax|54| logsumexp            | 55|  max                | 56 | min      |
| 57| mean      |58 | mm                   | 59| mul                 | 60 | neg      |
| 61| norm      | 62| ~~pixel_shuffle~~    | 63| pow                 | 64 | permute  |
| 65| ~~prelu~~ | 66| relu                 | 67| reshape             | 68 | relu6    |
| 69| softmax   | 70| slice                | 71| sub                 | 72 | sqrt     |
| 73| sigmoid   | 74| prod                 | 75| reshape_as          | 76 | round    |
| 77| rsqrt     | 78| ~~selu~~             | 79| sign                | 80 | sin      |
| 81| softplus  | 82| split                | 83| squeeze             | 84 | sum      |
| 85| tan       | 86| tanh                 | 87 | transpose          | 88 | unsqueeze|
| 89| ~~upsample_nearest2d~~ |

- hardswish pytorch导出存在bug
- interpolate oneflow和pytorch的参数列表未完全对齐，只能转nearest和align_corners=False的情况，working


## PaddlePaddle

| 序号 | OP | 序号 | OP | 序号 | OP | 序号 | OP |
|------|------|------|------|------|------|------|------|
| 1 | abs       | 2 |  acos              | 3 | add       | 4 |  argmax    |
| 5 | batch_norm| 6 | ~~bilinear_interp~~| 7 | bmm       | 8 |  ~~cast~~  |
| 9 | clip      | 10| concat             | 11| conv2d    | 12| ~~conv2d_transpose~~|
| 13| ~~cumsum~~| 14| depthwise_conv2d   | 15| dropout   | 16| elementwise_add|
| 17| elementwise_div| 18| elementwise_mul | 19| elementwise_min | 20| elementwise_max|
| 21| elementwise_pow| 22| elementwise_sub | 23| exp     | 24| expand_as  |
| 25| expand_dims|26| flatten            | 27| floor     | 28| gather     |
| 29| hardsigmoid|30| hardswish          | 31| leaky_relu| 32| log        |
| 33| matmul    | 34|    mean            | 35| mul       | 36| ~~nearest_interp~~|
| 37| pad2d     | 38| pow                | 39| ~~prelu~~ | 40| reduce_mean|
| 41| reduce_max| 42| reduce_min         | 43| reduce_prod|44| reduce_sum |
| 45| relu      | 46| relu6              | 47| reshape   | 48| softmax    |
| 49| sigmoid   | 50| slice              | 51| scale     | 52| ~~split~   |
| 53| squeeze   | 54| sqrt               | 55| square    | 56| stack      |
| 57| stride_slice|58| sum               | 59| swish     | 60| tanh       |
| 61| transpose | 62| unsqueeze| 


相关issue：

- https://github.com/PaddlePaddle/Paddle2ONNX/issues/221
- https://github.com/PaddlePaddle/Paddle2ONNX/issues/220
