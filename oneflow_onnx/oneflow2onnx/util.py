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
import tempfile
import numpy as np
import oneflow as flow
import onnxruntime as ort
from typing import Optional, Union, Tuple, List
from collections import OrderedDict
from oneflow_onnx.oneflow2onnx.flow2onnx import Export


def run_onnx(
    onnx_model_path: str,
    providers: List[str],
    ipt_dict: Optional[OrderedDict] = None,
    ort_optimize: bool = True,
) -> Union[Tuple[OrderedDict, np.ndarray], np.ndarray]:
    ort_sess_opt = ort.SessionOptions()
    ort_sess_opt.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if ort_optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    sess = ort.InferenceSession(
        onnx_model_path, sess_options=ort_sess_opt, providers=providers
    )
    assert len(sess.get_outputs()) == 1
    assert len(sess.get_inputs()) <= 1

    only_return_result = ipt_dict is not None

    if ipt_dict is None:
        ipt_dict = OrderedDict()
        for ipt in sess.get_inputs():
            ipt_data = np.random.uniform(low=-10, high=10, size=ipt.shape).astype(
                np.float32
            )
            ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)[0]

    if only_return_result:
        return onnx_res
    return ipt_dict, onnx_res


def export_onnx_model(
    graph,
    external_data=False,
    opset=None,
    flow_weight_dir=None,
    onnx_model_path="/tmp",
    dynamic_batch_size=False,
):
    if flow_weight_dir is None or os.path.exists(flow_weight_dir) == False:
        raise RuntimeError('Please specify the correct model path!')
    onnx_model_dir = onnx_model_path
    onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
    Export(
        graph,
        flow_weight_dir,
        onnx_model_path,
        opset=opset,
        external_data=external_data,
        dynamic_batch_size=dynamic_batch_size,
    )

    def cleanup():
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)

    return onnx_model_path, cleanup


def compare_result(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    print_outlier: bool = False,
):
    if print_outlier:
        a = a.flatten()
        b = b.flatten()
        for i in range(len(a)):
            if np.abs(a[i] - b[i]) > atol + rtol * np.abs(b[i]):
                print("a[{}]={}, b[{}]={}".format(i, a[i], i, b[i]))
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def convert_to_onnx_and_check(
    graph,
    print_outlier=False,
    explicit_init=False,
    external_data=False,
    ort_optimize=True,
    opset=None,
    flow_weight_dir=None,
    onnx_model_path="/tmp",
    dynamic_batch_size=False,
    device="cpu",
):
    onnx_model_path, cleanup = export_onnx_model(
        graph, external_data, opset, flow_weight_dir, onnx_model_path, dynamic_batch_size
    )


    if dynamic_batch_size != True:
        if ort.__version__>'1.9.0':
            ipt_dict, onnx_res = run_onnx(
            onnx_model_path, ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"], ort_optimize=ort_optimize
            )
        else:
            ipt_dict, onnx_res = run_onnx(
            onnx_model_path, ["CPUExecutionProvider"], ort_optimize=ort_optimize
            )
        if device=="gpu":
            oneflow_res = graph(flow.tensor(*ipt_dict.values(), dtype=flow.float32).to("cuda"))     
        else:
            oneflow_res = graph(flow.tensor(*ipt_dict.values(), dtype=flow.float32))
        if not isinstance(oneflow_res, np.ndarray):
            oneflow_res = oneflow_res.numpy()
        compare_result(oneflow_res, onnx_res, print_outlier=print_outlier)


    # cleanup()
