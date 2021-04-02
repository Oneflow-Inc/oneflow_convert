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

from __future__ import absolute_import
import setuptools

long_description = "oneflow_onnx is a toolkit for converting trained model of OneFlow to ONNX and ONNX to OneFlow.\n\n"
long_description += "Usage: oneflow_onnx --model_dir src --save_file dist\n"
long_description += "GitHub: https://github.com/Oneflow-Inc/oneflow_convert_tools/oneflow_onnx\n"
long_description += "Email: zhangxiaoyu@oneflow.org"

setuptools.setup(
    name="oneflow_onnx",
    version="0.1.0",
    author="zhangxiaoyu",
    author_email="zhangxiaoyu@oneflow.org",
    description="a toolkit for converting trained model of OneFlow to ONNX and ONNX to OneFlow.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/Oneflow-Inc/oneflow_convert_tools/oneflow_onnx",
    packages=setuptools.find_packages(),
    install_requires=['six', 'protobuf'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0'
)