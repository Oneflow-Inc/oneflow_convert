import oneflow as flow
import oneflow.nn as nn
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.zero = flow.zeros(1, 2, 3, 3).to("cuda")
        self.conv = nn.Conv2d(1, 2, 1, 1)

    def forward(self, x):
        return self.conv(x) + self.zero


m = ConvModule()
m.to("cuda")


class YOLOGraph(flow.nn.Graph):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def build(self, x):
        return self.model(x)


yolo_graph = YOLOGraph(m)
yolo_graph._compile(flow.randn(1, 1, 3, 3).to("cuda"))

convert_to_onnx_and_check(yolo_graph, onnx_model_path="/tmp", device="gpu")
