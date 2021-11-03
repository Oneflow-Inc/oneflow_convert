import os
import numpy as np

# import torch
# from torchvision import transforms, datasets
# import torchvision

import oneflow as flow
from flowvision import transforms, datasets
import flowvision
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

saved_file = "./oneflow_pretrained_model"

device = "cuda" if flow.cuda.is_available() else "cpu"
print("DEVICE USED:", device)

model_ft = flowvision.models.mobilenet_v2(pretrained=False)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = flow.nn.Sequential(
    flow.nn.Dropout(0.2), flow.nn.Linear(model_ft.last_channel, 40),
)

model_ft.to(device)
model_ft.eval()

class trashNetGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.m = model_ft

    def build(self, x):
        out = self.m(x)
        return out

if os.path.exists(saved_file):
    print("checkpoint file loading...")
    params = flow.load(saved_file)
    model_ft.load_state_dict(params)

trashNet_graph = trashNetGraph()

trashNet_graph._compile(flow.randn(1, 3, 224, 224).to(device))

convert_to_onnx_and_check(trashNet_graph, flow_weight_dir=saved_file, onnx_model_path="./")
