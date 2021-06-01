import os
import common
import argparse
import shutil
import numpy as np
import tensorrt as trt
import oneflow as flow
from models import get_lenet_job_function, LENET_MODEL_QAT_DIR
from oneflow_onnx.oneflow2onnx.util import export_onnx_model, run_onnx, compare_result


def build_engine_onnx(model_file, verbose=False):
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    )

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        flags=network_flags
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
        return builder.build_engine(network, config)


def run_tensorrt(onnx_path, test_case):
    with build_engine_onnx(onnx_path) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            batch_size = test_case.shape[0]
            test_case = test_case.reshape(-1)
            np.copyto(inputs[0].host, test_case)
            trt_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            data = trt_outputs[0]
            return data.reshape(batch_size, -1)


def test_lenet_qat():
    batch_size = 16
    predict_job = get_lenet_job_function("predict", batch_size=batch_size)
    flow.load_variables(flow.checkpoint.get(LENET_MODEL_QAT_DIR))

    onnx_model_path, cleanup = export_onnx_model(predict_job,opset=13)

    ipt_dict, onnx_res = run_onnx(onnx_model_path)
    oneflow_res = predict_job(*ipt_dict.values())
    compare_result(oneflow_res, onnx_res)

    trt_res = run_tensorrt(onnx_model_path, ipt_dict[list(ipt_dict.keys())[0]])
    compare_result(oneflow_res, trt_res, True)

    flow.clear_default_session()
    cleanup()
    if os.path.exists(LENET_MODEL_QAT_DIR):
        shutil.rmtree(LENET_MODEL_QAT_DIR)

test_lenet_qat()
