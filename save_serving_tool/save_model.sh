#!/bin/bash
set -ex

# download model parameters for first-time
# wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/cpu/resnet50.tar.gz
# tar zxvf resnet50.tar.gz

base_dir=`dirname $0`

python3 $base_dir/save_model.py \
    --model_dir resnet50_nhwc \
    --save_dir resnet50_models \
    --model_version 1 \
    --force_save
