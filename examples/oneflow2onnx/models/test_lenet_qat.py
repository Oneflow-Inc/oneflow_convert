import os
import numpy as np
import oneflow as flow
from models import get_lenet_job_function
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

def test_lenet_qat():
    batch_size = 100
    predict_job = get_lenet_job_function("predict", batch_size=batch_size)
    temp_dir_name = ""
    with open("lenet_qat_temp_dir_name.txt","r") as f:
        temp_dir_name = f.readline()
    temp_dir = os.path.join("/tmp", temp_dir_name)
    convert_to_onnx_and_check(predict_job, flow_weight_dir=temp_dir, onnx_model_path="/tmp")
