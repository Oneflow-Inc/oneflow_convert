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
import numpy as np
import oneflow as flow
import onnxruntime as ort
from typing import Optional, Union, Tuple, List
from collections import OrderedDict
from oneflow_onnx.oneflow2onnx.flow2onnx import Export

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def run_onnx(
    onnx_model_path: str, providers: List[str], ipt_dict: Optional[OrderedDict] = None, ort_optimize: bool = True, input_tensor_range: List = None,
) -> Union[Tuple[OrderedDict, np.ndarray], np.ndarray]:
    ort_sess_opt = ort.SessionOptions()
    ort_sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if ort_optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(onnx_model_path, sess_options=ort_sess_opt, providers=providers)

    only_return_result = ipt_dict is not None

    if ipt_dict is None:
        ipt_dict = OrderedDict()
        for ipt in sess.get_inputs():
            low, high = -10, 10
            if input_tensor_range is not None:
                low = input_tensor_range[0]
                high = input_tensor_range[1]
            ipt_data = np.random.uniform(low=low, high=high, size=ipt.shape)
            if ipt.type == "tensor(int64)":
                ipt_data = ipt_data.astype(np.int64)
            elif ipt.type == "tensor(float)":
                ipt_data = ipt_data.astype(np.float32)
            elif ipt.type == "tensor(double)":
                ipt_data = ipt_data.astype(np.float64)
            elif ipt.type == "tensor(bool)":
                ipt_data = ipt_data.astype(np.bool)
            else:
                raise NotImplementedError(f"{ipt.type} is not supported now, please give a feedback in https://github.com/Oneflow-Inc/oneflow_convert/issues/new .")
            ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)[0]

    if only_return_result:
        return onnx_res
    return ipt_dict, onnx_res


def export_onnx_model(
    graph, external_data=False, opset=None, flow_weight_dir=None, onnx_model_path="/tmp", dynamic_batch_size=False,
):
    flow_weight_clean_flag = False
    if flow_weight_dir is None:
        flow_weight_clean_flag = True
        flow_weight_dir = os.path.join("/tmp/", flow._oneflow_internal.UniqueStr("oneflow_model"))
        if os.path.exists(flow_weight_dir):
            shutil.rmtree(flow_weight_dir)
        if graph._is_global_view:
            # save global tensor model
            flow.save(graph.state_dict(), flow_weight_dir, global_dst_rank=0)
        else:
            # save local tensor model
            flow.save(graph.state_dict(), flow_weight_dir)

    onnx_model_dir = onnx_model_path
    if os.path.isdir(onnx_model_path):
        onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
    print("Converting model to onnx....")
    Export(
        graph, flow_weight_dir, onnx_model_path, opset=opset, external_data=external_data, dynamic_batch_size=dynamic_batch_size,
    )
    print(f"Succeed converting model, save model to {onnx_model_path}")

    def cleanup():
        if os.path.exists(flow_weight_dir) and flow_weight_clean_flag:
            shutil.rmtree(flow_weight_dir)

    return onnx_model_path, cleanup


def compare_result(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-2, atol: float = 1e-5, print_outlier: bool = False,
):
    if print_outlier:
        a = a.flatten()
        b = b.flatten()
        for i in range(len(a)):
            if np.abs(a[i] - b[i]) > atol + rtol * np.abs(b[i]):
                print("a[{}]={}, b[{}]={}".format(i, a[i], i, b[i]))
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def convert_to_onnx_and_check(
    graph, print_outlier=True, external_data=False, ort_optimize=True, opset=None, flow_weight_dir=None, onnx_model_path="/tmp", dynamic_batch_size=False, device="cpu", input_tensor_range=None,
):
    onnx_model_path, cleanup = export_onnx_model(graph, external_data, opset, flow_weight_dir, onnx_model_path, dynamic_batch_size,)

    if input_tensor_range is not None:
        assert isinstance(input_tensor_range, List), f"input_tensor_range {input_tensor_range} must be a List, e.g. [-5, 5]"
        assert len(input_tensor_range) == 2 and input_tensor_range[0] < input_tensor_range[1], f"input_tensor_range {input_tensor_range} must be a increasing List with two elements, e.g. [0, 10]"

    if dynamic_batch_size != True:
        if ort.__version__ > "1.9.0":
            ipt_dict, onnx_res = run_onnx(
                onnx_model_path, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider",], ort_optimize=ort_optimize, input_tensor_range=input_tensor_range,
            )
        else:
            ipt_dict, onnx_res = run_onnx(onnx_model_path, ["CPUExecutionProvider"], ort_optimize=ort_optimize, input_tensor_range=input_tensor_range,)

        oneflow_res = None

        if device == "gpu":
            device_kwargs = dict(device="cuda")
        elif device == "cpu":
            device_kwargs = dict(device="cpu")
        elif device == "gpu_global":
            device_kwargs = dict(sbp=flow.sbp.broadcast, placement=flow.placement("cuda", ranks=[0]))
        elif device == "cpu_global":
            device_kwargs = dict(sbp=flow.sbp.broadcast, placement=flow.placement("cpu", ranks=[0]))
        else:
            raise NotImplementedError

        if len(ipt_dict) == 0:
            oneflow_res = graph()
        else:
            graph_input_tensor = []
            for _, value in ipt_dict.items():
                value_tensor = None
                if str(value.dtype) == "int64":
                    value_tensor = flow.tensor(value, dtype=flow.int64, **device_kwargs)
                elif str(value.dtype) == "float" or str(value.dtype) == "float32":
                    value_tensor = flow.tensor(value, dtype=flow.float32, **device_kwargs)
                elif str(value.dtype) == "float64":
                    value_tensor = flow.tensor(value, dtype=flow.float64, **device_kwargs)
                elif str(value.dtype) == "bool":
                    value_tensor = flow.tensor(value, dtype=flow.bool, **device_kwargs)
                else:
                    raise NotImplementedError(f"{value.dtype} is not supported now, please give a feedback in https://github.com/Oneflow-Inc/oneflow_convert/issues/new .")
                graph_input_tensor.append(value_tensor)

            try:
                oneflow_res = graph(graph_input_tensor)
            except:
                print(
                    f"\033[0;36mInput Tensor or Weight by nn.Graph complied is not in Eager Local mode, maybe in Eager Global mode? In Eager Local Mode we can not compare result diffrience, so the inference result of the onnx model maybe not correct. We strongly recommend that you export onnx in Eager Local mode!\033[0;36m"
                )

        if oneflow_res is not None:
            if not isinstance(oneflow_res, np.ndarray):
                if flow.is_tensor(oneflow_res):
                    pass
                elif isinstance(oneflow_res, dict):
                    for key, value in oneflow_res.items():
                        oneflow_res = value
                        break
                elif isinstance(oneflow_res, (list, tuple)):
                    oneflow_res = oneflow_res[0]
                else:
                    raise NotImplementedError
            if flow.is_tensor(oneflow_res):
                if "global" in device:
                    oneflow_res = oneflow_res.to_local()
                oneflow_res = oneflow_res.numpy()
            print("Comparing result between oneflow and onnx....")
            compare_result(oneflow_res, onnx_res, print_outlier=print_outlier)
            print("Compare succeed!")
    # cleanup()
