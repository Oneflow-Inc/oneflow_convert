## oneflow_convert_tools


### oneflow_onnx

[![PyPI version](https://img.shields.io/pypi/v/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI license](https://img.shields.io/pypi/l/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/oneflow_convert_tools/pulls)

#### Introduction

- OneFlow2ONNX models are supported. Specifically, OneFlow's eager mode model can be transfomed into ONNX's format. For more information, please refer to [OneFlow2ONNX Model List](docs/oneflow2onnx/oneflow2onnx_model_zoo.md).
- OneFlow2ONNX operators are supported. Please refer to [OneFlow2ONNX Operator Lists](docs/oneflow2onnx/op_list.md) for more information.


#### Environment Dependencies

##### User's Environment Configuration

```sh
python>=3.5
onnx>=1.8.0
onnxruntime>=1.6.0
oneflow (https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)
```


#### Installation

##### Method 1

```sh
pip install oneflow_onnx
```

**Method 2**

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

#### Usage

Please refer to [Examples](examples/README.md)

#### Related Documents

- [OneFlow2ONNX Model List](docs/oneflow2onnx/oneflow2onnx_model_zoo.md)
- [OneFlow2ONNX Operator List](docs/oneflow2onnx/op_list.md)
- [Examples](examples/README.md)

