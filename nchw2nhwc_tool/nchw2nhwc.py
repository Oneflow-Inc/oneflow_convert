#coding=utf-8

import os
import argparse
import numpy as np
import oneflow as flow
import shutil

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
    new_m = os.path.join(output_model_dir, file)
    if (os.path.isdir(m)):
        weight_file = os.path.join(m, "out")
        meta_file = os.path.join(m, "meta")
        
        new_weight_file = os.path.join(new_m, "out")
        new_meta_file = os.path.join(new_m, "meta")
        
        import subprocess
        tensor_shape = subprocess.check_output(("grep dim {} | sed 's/dim: //g' | xargs").format(meta_file), shell=True)
        
        dims = list(map(int, str(tensor_shape, encoding = "utf8").strip().split()))

        if len(dims)  == 4:
            print(dims)
            # [n, c, h, w] - > [n, h, w, c]
            # [0, 1, 2, 3] -> [0, 2, 3, 1]
            weight = np.fromfile(weight_file, count)
            print(weight.shape)
            weight = weight.reshape(dims)
            weight = np.transpose(weight, (0, 2, 3, 1))
        if not os.path.exists(new_weight_file):
            os.mkdir(new_weight_file)
        if not os.path.exists(new_meta_file):
            os.mkdir(new_meta_file)
        
        f = open(new_weight_file, 'wb')
        f.write(weight)
        f.close()
        shutil.copy(meta_file, new_meta_file)
        
    elif (m == "snapshot_done"):
        if not os.path.exists(new_m):
            os.mkdir(new_m)
        shutil.copy(m, new_m)
    else:
        pass
        




