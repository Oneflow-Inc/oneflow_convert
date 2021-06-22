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
from collections import OrderedDict
import tempfile
import os
import shutil

import numpy as np
import onnxruntime as ort
import onnx
import torch
import paddle
import tensorflow as tf

import oneflow as flow
import oneflow.typing as tp
from oneflow_onnx.x2oneflow.handler import oneflow_code_gen, oneflow_blobname_map
from oneflow_onnx.x2oneflow.onnx2flow import from_onnx, from_pytorch, from_paddle, from_tensorflow2

def oneflow_code_gen_func(input_size, model_weight_save_dir):
    oneflow_python_file = "/tmp/oneflow_code.py"
    onnx_file = '/tmp/simp.onnx'
    f = open(oneflow_python_file, 'w')

    f.write('import oneflow as flow\n')
    f.write('import oneflow.typing as tp\n')
    f.write('import numpy as np\n')
    f.write('import onnxruntime as ort\n')
    f.write('from collections import OrderedDict\n\n')

    f.write('@flow.global_function(type="predict")\n')
    f.write('def eval_job(\n')
    f.write('   x_0: tp.Numpy.Placeholder(({}, {}, {}, {}), dtype=flow.float)\n'.format(input_size[0], input_size[1], input_size[2], input_size[3]))
    f.write(') -> tp.Numpy:\n')
    f.write('   with flow.scope.placement("gpu", "0:0"):\n')



    for x in oneflow_code_gen:
        f.write('     {}'.format(x))
    
    res = oneflow_code_gen[len(oneflow_code_gen)-1].split()[0]
    f.write('     return {}'.format(res))

    f.write('\n\n')
    
    f.write('def main():\n')
    f.write('   x = np.random.uniform(low=0.0, high=1.0, size=({}, {}, {}, {})).astype(np.float32)\n'.format(input_size[0], input_size[1], input_size[2], input_size[3]))
    f.write('   flow.train.CheckPoint().load({})\n'.format("'"+model_weight_save_dir+"'"))
    f.write('   oneflow_res = eval_job(x)\n\n')
    f.write('   ort_sess_opt = ort.SessionOptions()\n')
    f.write('   ort_sess_opt.graph_optimization_level = (\n')
    f.write('     ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED\n')
    f.write('   )\n\n')
    f.write('   sess = ort.InferenceSession({}, sess_options=ort_sess_opt)\n'.format("'"+onnx_file+"'"))
    f.write('   assert len(sess.get_outputs()) == 1\n')
    f.write('   assert len(sess.get_inputs()) <= 1\n')
    f.write('   ipt_dict = OrderedDict()\n')
    f.write('   for ipt in sess.get_inputs():\n')
    f.write('     ipt_dict[ipt.name] = x\n\n')
    f.write('   onnx_res = sess.run([], ipt_dict)[0]\n')
    f.write('   rtol, atol = 1e-2, 1e-5\n')
    f.write('   a = onnx_res.flatten()\n')
    f.write('   b = oneflow_res.flatten()\n')
    f.write('   for i in range(len(a)):\n')
    f.write('     if np.abs(a[i] - b[i]) > atol + rtol * np.abs(b[i]):\n')
    f.write('        print("a[{}]={}, b[{}]={}".format(i, a[i], i, b[i]))\n\n')
    f.write('   assert np.allclose(onnx_res, oneflow_res, rtol=rtol, atol=atol)\n\n')
    
    f.write('if __name__ == "__main__":\n')
    f.write('   main()\n')
    f.close()

def load_pytorch_module_and_check(
    pt_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    flow_weight_dir="/tmp/oneflow",
    oneflow_code_gen_flag=False,
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pt_module = pt_module_class()

    model_weight_save_dir = flow_weight_dir

    if train_flag == True:

        @flow.global_function(type="train")
        def job_train(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_pytorch(
                pt_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0])
            flow.optimizer.SGD(lr_scheduler).minimize(y)
            return y

    else:

        @flow.global_function(type="predict")
        def job_eval(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_pytorch(
                pt_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            return y

    flow.train.CheckPoint().load(model_weight_save_dir)
    # flow.load_variables(flow.checkpoint.get(model _weight_save_dir))

    if oneflow_code_gen_flag == True and len(input_size) == 4:
        oneflow_code_gen_func(input_size, model_weight_save_dir)
        flow.clear_default_session()
        return
        
    if train_flag == False:
        pt_module.eval()
    
    ipt1 = np.random.uniform(
        low=input_min_val, high=input_max_val, size=input_size
    ).astype(np.float32)

    if train_flag == True:
        flow_res = job_train(ipt1)
    else:
        flow_res = job_eval(ipt1)
    pytorch_res = pt_module(torch.tensor(ipt1).to("cpu")).detach().numpy()
    print(flow_res)
    print("-------------")
    print(pytorch_res)

    a, b = flow_res.flatten(), pytorch_res.flatten()

    max_idx = np.argmax(np.abs(a - b) / (a + 1e-7))
    print(
        "max rel diff is {} at index {}".format(
            np.max(np.abs(a - b) / (a + 1e-7)), max_idx
        )
    )
    print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    flow.clear_default_session()


def load_paddle_module_and_check(
    pd_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    flow_weight_dir="/tmp/oneflow",
    oneflow_code_gen_flag = False, 
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pd_module = pd_module_class()

    model_weight_save_dir = flow_weight_dir

    if train_flag == True:

        @flow.global_function(type="train")
        def job_train(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_paddle(
                pd_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0])
            flow.optimizer.SGD(lr_scheduler).minimize(y)
            return y

    else:

        @flow.global_function(type="predict")
        def job_eval(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_paddle(
                pd_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            return y

    flow.train.CheckPoint().load(model_weight_save_dir)

    if oneflow_code_gen_flag == True and len(input_size) == 4:
        oneflow_code_gen_func(input_size, model_weight_save_dir)
        flow.clear_default_session()
        return
    
    if train_flag == False:
        pd_module.eval()

    ipt1 = np.random.uniform(
        low=input_min_val, high=input_max_val, size=input_size
    ).astype(np.float32)
    if train_flag == True:
        flow_res = job_train(ipt1)
    else:
        flow_res = job_eval(ipt1)
    paddle_res = pd_module(paddle.to_tensor(ipt1)).numpy()
    print(flow_res)
    print("-------------")
    print(paddle_res)

    a, b = flow_res.flatten(), paddle_res.flatten()

    max_idx = np.argmax(np.abs(a - b) / (a + 1e-7))
    print(
        "max rel diff is {} at index {}".format(
            np.max(np.abs(a - b) / (a + 1e-7)), max_idx
        )
    )
    print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    flow.clear_default_session()


def load_tensorflow2_module_and_check(
    tf_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    flow_weight_dir="/tmp/oneflow",
    oneflow_code_gen_flag = False, 
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    tf_module = tf_module_class()
    
    # flow.config.enable_debug_mode(True)

    model_weight_save_dir = flow_weight_dir

    if train_flag == True:

        @flow.global_function(type="train")
        def job_train(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_tensorflow2(
                tf_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0])
            flow.optimizer.SGD(lr_scheduler).minimize(y)
            return y

    else:

        @flow.global_function(type="predict")
        def job_eval(x: tp.Numpy.Placeholder(input_size)) -> tp.Numpy:
            x += flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )

            y = from_tensorflow2(
                tf_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )
            return y

    flow.train.CheckPoint().load(model_weight_save_dir)

    ipt1 = np.random.uniform(
        low=input_min_val, high=input_max_val, size=input_size
    ).astype(np.float32)

    if oneflow_code_gen_flag == True and len(input_size) == 4:
        oneflow_code_gen_func(input_size, model_weight_save_dir)
        flow.clear_default_session()
        return
    
    if train_flag == True:
        flow_res = job_train(ipt1)
    else:
        flow_res = job_eval(ipt1)

    tf_input = tf.constant(ipt1, dtype=tf.float32)
    tensorflow_res = tf_module.predict(tf_input)
    if type(tensorflow_res) is not list:
        tensorflow_res = np.array(tensorflow_res)

    print(flow_res)
    print("-------------")
    print(tensorflow_res)

    a, b = flow_res.flatten(), tensorflow_res.flatten()

    max_idx = np.argmax(np.abs(a - b) / (a + 1e-7))
    print(
        "max rel diff is {} at index {}".format(
            np.max(np.abs(a - b) / (a + 1e-7)), max_idx
        )
    )
    print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    flow.clear_default_session()

