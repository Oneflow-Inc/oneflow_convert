#coding=utf-8

import os
import argparse
import numpy as np
import oneflow as flow

parser = argparse.ArgumentParser()

parser.add_argument("--input_model_dir", type=str,
                        default="./resnet50", help="input model directory")

parser.add_argument("--output_model_dir", type=str,
                        default="./resnet50_nhwc", help="output model directory")

args = parser.parse_args()

input_model_dir = args.input_model_dir
output_model_dir = args.output_model_dir

files = os.listdir(input_model_dir)

for file in files:
    m = os.path.join(input_model_dir, file)
    if (os.path.isdir(m)):
        weight_file = os.path.join(m, "out")
        meta_file = os.path.join(m, "meta")
        
        import subprocess
        tensor_shape = subprocess.check_output(("grep dim {} | sed 's/dim: //g' | xargs").format(meta_file), shell=True)
        
        dims = list(map(int, str(tensor_shape, encoding = "utf8").strip().split()))

        if len(dims)  == 4:
            print(dims)




