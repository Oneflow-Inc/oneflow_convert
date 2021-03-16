#coding=utf-8

import os
import argparse
import numpy as np
import oneflow as flow
import shutil
import struct

parser = argparse.ArgumentParser()

parser.add_argument("--input_model_dir", type=str,
                        default="./resnet50", help="input model directory")

parser.add_argument("--output_model_dir", type=str,
                        default="./resnet50_nhwc", help="output model directory")

args = parser.parse_args()

input_model_dir = args.input_model_dir
output_model_dir = args.output_model_dir

files = os.listdir(input_model_dir)

if os.path.exists(output_model_dir):
    del_list = os.listdir(output_model_dir)
    for f in del_list:
        file_path = os.path.join(output_model_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    os.rmdir(output_model_dir)

for file in files:
    m = os.path.join(input_model_dir, file)
    new_m = os.path.join(output_model_dir, file)

    if (os.path.isdir(m)):
        if not os.path.exists(new_m):
            os.makedirs(new_m)
        
        weight_file = os.path.join(m, "out")
        meta_file = os.path.join(m, "meta")
        
        new_weight_file = os.path.join(new_m, "out")
        new_meta_file = os.path.join(new_m, "meta")
        
        import subprocess
        tensor_shape = subprocess.check_output(("grep dim {} | sed 's/dim: //g' | xargs").format(meta_file), shell=True)
        
        dims = list(map(int, str(tensor_shape, encoding = "utf8").strip().split()))

        if len(dims)  == 4:
            print(dims)
            weight = []
            # [n, c, h, w] - > [n, h, w, c]
            # [0, 1, 2, 3] -> [0, 2, 3, 1]
            binfile = open(weight_file, 'rb')
            size = os.path.getsize(weight_file)
            for i in range(size // 4):
                data = binfile.read(4)
                weight.append(struct.unpack('f', data))
            
            weight = np.array(weight)
            weight = weight.reshape(dims)
            weight = np.transpose(weight, (0, 2, 3, 1))
        
        os.mknod(new_weight_file)
        os.mknod(new_meta_file)
        
        f = open(new_weight_file, 'wb')
        f.write(np.ascontiguousarray(weight))
        f.close()
        shutil.copy(meta_file, new_meta_file)
        
    elif (m == "snapshot_done"):
        os.mknod(new_m)
        shutil.copy(m, new_m)
    else:
        pass
        




