# OneFlow2ONNX 支持的OP列表

> 目前OneFlow2ONNX 支持90+的ONNX OP，我们在下面的列表中列出了目前OneFlow支持导出的全部ONNX OP。


| 序号 | OP         | 序号 | OP             | 序号 | OP          | 序号 | OP                 |
| ---- | ---------- | ---- | -------------- | ---- | ----------- | ---- | ------------------ |
| 1    | GatherND   | 2    | Transpose      | 3    | Add         | 4    | Sub                |
| 5    | Mul        | 6    | Div            | 7    | Sum         | 8    | LeakyRelu          |
| 9    | Softplus   | 10   | Softplus       | 11   | Abs         | 12   | Ceil               |
| 13   | Elu        | 14   | Exp            | 15   | Floor       | 16   | Log                |
| 17   | Neg        | 18   | Sigmoid        | 19   | Sqrt        | 20   | Tanh               |
| 21   | Reciprocal | 22   | Relu           | 23   | Acos        | 24   | Asin               |
| 25   | Atan       | 26   | Cos            | 27   | Sin         | 28   | Tan                |
| 29   | Acosh      | 30   | Asinh          | 31   | Atanh       | 32   | Cosh               |
| 33   | Sinh       | 34   | Min            | 35   | Max         | 36   | Clip               |
| 37   | Softmax    | 38   | Sign           | 39   | MatMul      | 40   | Erf                |
| 41   | FloorMod   | 42   | Round          | 43   | Not         | 44   | And                |
| 45   | Or         | 46   | Equal          | 47   | NotEqual    | 48   | Greater            |
| 49   | Less       | 50   | Pad            | 51   | AveragePool | 52   | MaxPool            |
| 53   | Conv       | 54   | QuantizeLinear | 56   | ReduceMin   | 57   | BatchNormalization |
| 58   | ReduceSum  | 59   | ReduceProd     | 60   | ArgMax      | 61   | ArgMin             |
| 62   | Reshape    | 63   | Squeeze        | 64   | Transpose   | 65   | Concat             |
| 66   | Cast       | 67   | Identity       | 68   | Mul         | 69   | PReLU              |
| 70   | LeakyReLU  | 71   | Constant       | 72   | Flatten     | 73   | Slice              |
| 74   | Pooling    | 75   | Groupconvd     | 76   | HardSwish   | 77   | HardSigmoid        |
| 78   | Arange     | 79   | ExpandDims     | 80   | Narrow      | 81   | SiLU               |
| 82   | Upsample   | 83   | Var            | 84   | Conv1D      | 85   | ScalarDiv          |
| 86   | CublasFusedMLP| 87| Unsqueeze      | 88   | BroadcastMatmul | 89| Where             |
| 90   | ScalarLogicalLess| 91| ScalarLogicalGreater| 92| Gather  | 93  | Expand             |
| 94   | fill_      | 95   | GeLU           | 96   | LayerNorm    | 97  | AmpIdentity        |
| 98   | fast_gelu  | 99   | quick_gelu     | 100  | fused_self_attention |101 |RMSLayerNorm |
