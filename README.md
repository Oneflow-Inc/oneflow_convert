## oneflow_convert_tools

OneFlow相关的模型转换工具

### onnx

#### 简介

onnx工具包含两个功能，一个是将OneFlow导出ONNX，另外一个是将各个训练框架导出的ONNX模型转换为OneFlow的模型。本工程已经适配了TensorFlow/Pytorch/Paddle框架的预训练模型通过导出ONNX转换为OneFlow（我们将这一功能叫作X2OneFlow）。

#### 环境依赖

##### 用户环境配置

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
oneflow>=0.3.4
```

#### 安装

##### 安装方式1

```sh
pip install oneflow_onnx
```

#### 安装方式2

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

```sh
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
python setup.py install
```

### nchw2nhwc_tool

#### 简介

本工具的功能是将OneFlow训练的NCHW排布的权重转换为NHWC排布，使用方法[在这里](nchw2nhwc_tool/README.md)

