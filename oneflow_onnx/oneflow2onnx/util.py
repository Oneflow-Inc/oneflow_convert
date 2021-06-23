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
from collections import OrderedDict
from oneflow_onnx.oneflow2onnx.flow2onnx import Export


def run_onnx(onnx_model_path, ort_optimize=True):
    ort_sess_opt = ort.SessionOptions()
    ort_sess_opt.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if ort_optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    sess = ort.InferenceSession(onnx_model_path, sess_options=ort_sess_opt)
    assert len(sess.get_outputs()) == 1
    assert len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_data = np.random.uniform(low=-10, high=10, size=ipt.shape).astype(
            np.float32
        )
        ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)[0]
    return ipt_dict, onnx_res


def export_onnx_model(
    job_func,
    external_data=False,
    opset=None,
    flow_weight_dir=None,
    onnx_model_path="/tmp",
):
    if flow_weight_dir == None:
        flow_weight_dir = tempfile.TemporaryDirectory()
        flow.checkpoint.save(flow_weight_dir.name)
        # TODO(daquexian): a more elegant way?
        while not os.path.exists(os.path.join(flow_weight_dir.name, "snapshot_done")):
            pass
        onnx_model_dir = onnx_model_path
        onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
        Export(
            job_func,
            flow_weight_dir.name,
            onnx_model_path,
            opset=opset,
            external_data=external_data,
        )
        flow_weight_dir.cleanup()
    else:
        while not os.path.exists(os.path.join(flow_weight_dir, "snapshot_done")):
            pass
        onnx_model_dir = onnx_model_path
        onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
        Export(
            job_func,
            flow_weight_dir,
            onnx_model_path,
            opset=opset,
            external_data=external_data,
        )

    def cleanup():
        if os.path.exists(onnx_model_path):
            os.remove(onnx_model_path)

    return onnx_model_path, cleanup


def compare_result(a, b, print_outlier=False):
    rtol, atol = 1e-2, 1e-5
    if print_outlier:
        a = a.flatten()
        b = b.flatten()
        for i in range(len(a)):
            if np.abs(a[i] - b[i]) > atol + rtol * np.abs(b[i]):
                print("a[{}]={}, b[{}]={}".format(i, a[i], i, b[i]))
    assert np.allclose(a, b, rtol=rtol, atol=atol)


def convert_to_onnx_and_check(
    job_func,
    print_outlier=False,
    explicit_init=False,
    external_data=False,
    ort_optimize=True,
    opset=None,
    flow_weight_dir=None,
    onnx_model_path="/tmp",
):
    if explicit_init:
        # it is a trick to keep check_point.save() from hanging when there is no variable
        @flow.global_function()
        def add_var():
            return flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
            )

    flow.train.CheckPoint().init()

    onnx_model_path, cleanup = export_onnx_model(
        job_func, external_data, opset, flow_weight_dir, onnx_model_path
    )

    ipt_dict, onnx_res = run_onnx(onnx_model_path, ort_optimize)
    oneflow_res = job_func(*ipt_dict.values())
    if not isinstance(oneflow_res, np.ndarray):
        oneflow_res = oneflow_res.get().numpy()

    compare_result(oneflow_res, onnx_res, print_outlier)

    flow.clear_default_session()
    # cleanup()
