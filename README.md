## oneflow_convert_tools

**[简体中文](README.md) | [English](README_en.md)**

OneFlow 相关的模型转换工具

### oneflow_onnx

[![PyPI version](https://img.shields.io/pypi/v/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI license](https://img.shields.io/pypi/l/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/oneflow_convert_tools/pulls)

#### 简介


- OneFlow2ONNX 模型支持，支持 OneFlow 静态图模型转为 ONNX，可转换由 [flow.checkpoint.save ](https://docs.oneflow.org/basics_topics/model_load_save.html) 方法保存下来的 OneFlow 模型，详情可以参考 [OneFlow2ONNX 模型列表](docs/oneflow2onnx/oneflow2onnx_model_zoo.md)。
- OneFlow2ONNX 算子支持，目前稳定支持导出 ONNX Opset10，部分 OneFlow 算子支持更低的 ONNX Opset 转换，详情可以参考 [OneFlow2ONNX 算子列表](docs/oneflow2onnx/op_list.md)。


#### 环境依赖

##### 用户环境配置

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
oneflow>=0.5.0
```

#### 安装

##### 安装方式1

```sh
pip install oneflow_onnx
```

**安装方式2**

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

#### 使用方法

请参考[使用示例](examples/README.md)

#### 相关文档

- [OneFlow2ONNX模型列表](docs/oneflow2onnx/oneflow2onnx_model_zoo.md)
- [OneFlow2ONNX算子列表](docs/oneflow2onnx/op_list.md)
- [使用示例](examples/README.md)




