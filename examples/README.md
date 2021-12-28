[简体中文](README_zh.md) | English

## oneflow_onnx samples

At present, oneflow2onnx supports the export of 80 + OneFlow OP to ONNX Op.

### export_onnx_model(`graph, external_data=False, opset=None, flow_weight_dir=None, onnx_model_path="/tmp", dynamic_batch_size=False`)

**PARAMETERS:**

 - graph - The graph to be converted (`oneflow.nn.Graph`).
 - external_data - Save weights as ONNX external data, usually to bypass the 2GB file size limit of protobuf.
 - opset - Specifies the version of the transformation model. The opset to be used (`int`, default is `oneflow_onnx.constants.PREFERRED_OPSET`)
 - flow_weight_dir - Path to save neflow model weights.
 - onnx_model_path - The directory containing OneFlow model weights. The user needs to call `check_point.save(flow_weight_dir)` and wait for the OneFlow model weight to be saved.
 - dynamic_batch_size - Does the converted onnx model support dynamic batch. The default value is `False`.