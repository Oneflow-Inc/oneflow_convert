## oneflow_onnx 使用示例

目前OneFlow2ONNX 支持80+的OneFlow OP导出为ONNX OP。

### export_onnx_model(`graph, external_data=False, opset=None, flow_weight_dir=None, onnx_model_path="/tmp", dynamic_batch_size=False`)

**参数列表:**

 - graph - 需要转换的graph(`oneflow.nn.Graph`)。
 - external_data - 将权重另存为ONNX模型的外部数据，通常是为了避免protobuf的2GB文件大小限制。
 - opset - 指定转换模型的版本(`int`，默认为`oneflow_onnx.constants.PREFERRED_OPSET`)。
 - flow_weight_dir - OneFlow模型权重的保存路径。
 - onnx_model_path - 导出的ONNX模型保存路径。用户需要调用`check_point.save(flow_weight_dir)`，等待OneFlow模型权重保存完成。
 - dynamic_batch_size - 转化的ONNX模型是否支持动态Batch，默认为`False`。