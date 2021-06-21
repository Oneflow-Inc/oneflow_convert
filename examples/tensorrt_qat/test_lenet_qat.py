"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import shutil
import oneflow as flow
from common import run_tensorrt, get_onnx_provider
from models import get_lenet_job_function, LENET_MODEL_QAT_DIR
from oneflow_onnx.oneflow2onnx.util import export_onnx_model, run_onnx, compare_result


def test_lenet_qat():
    model_existed = os.path.exists(LENET_MODEL_QAT_DIR)
    assert model_existed
    # Without the following 'print' CI won't pass, but I have no idea why.
    print("Model exists. " if model_existed else "Model does not exist. ")

    batch_size = 32
    predict_job = get_lenet_job_function("predict", batch_size=batch_size)
    flow.load_variables(flow.checkpoint.get(LENET_MODEL_QAT_DIR))

    onnx_model_path, cleanup = export_onnx_model(predict_job, opset=10)

    ipt_dict, onnx_res = run_onnx(onnx_model_path, get_onnx_provider("cpu"))
    oneflow_res = predict_job(*ipt_dict.values())
    compare_result(oneflow_res, onnx_res, print_outlier=True)

    trt_res = run_tensorrt(onnx_model_path, ipt_dict[list(ipt_dict.keys())[0]])
    compare_result(oneflow_res, trt_res, print_outlier=True)

    flow.clear_default_session()
    cleanup()
    if os.path.exists(LENET_MODEL_QAT_DIR):
        shutil.rmtree(LENET_MODEL_QAT_DIR)
