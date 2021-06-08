import os
import shutil
import oneflow as flow
from common import run_tensorrt
from models import get_mobilenet_job_function, MOBILENET_MODEL_QAT_DIR
from oneflow_onnx.oneflow2onnx.util import export_onnx_model, run_onnx, compare_result


def test_mobilenet_qat():
    # Without the following 'print' CI won't pass, but I have no idea why.
    print(
        "Model exists. "
        if os.path.exists(MOBILENET_MODEL_QAT_DIR)
        else "Model does not exist. "
    )
    batch_size = 4
    predict_job = get_mobilenet_job_function("predict", batch_size=batch_size)
    flow.load_variables(flow.checkpoint.get(MOBILENET_MODEL_QAT_DIR))
    onnx_model_path, cleanup = export_onnx_model(predict_job, opset=10)

    ipt_dict, onnx_res = run_onnx(onnx_model_path)
    oneflow_res = predict_job(*ipt_dict.values())
    compare_result(oneflow_res, onnx_res, print_outlier=True)

    trt_res = run_tensorrt(onnx_model_path, ipt_dict[list(ipt_dict.keys())[0]])
    compare_result(oneflow_res, trt_res, True)

    flow.clear_default_session()
    cleanup()
    if os.path.exists(MOBILENET_MODEL_QAT_DIR):
        shutil.rmtree(MOBILENET_MODEL_QAT_DIR)
