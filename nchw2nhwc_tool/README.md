### nchw2nhwc_tool

#### 依赖
- oneflow

#### 执行指令

- 模型转换

```
python3 nchw2nhwc.py --input_model_dir="./resnet50" --output_model_dir="./resnet50_nhwc"
```

- 模型推理

```
对于NCHW：python3 inference.py --log_dir="./log" --model_load_dir="./resnet50" --image_path="./fish.jpg" --channel_last=False

对于NHWC: python3 inference.py --log_dir="./log" --model_load_dir="./resnet50_nhwc" --image_path="./fish.jpg" --channel_last=True
```

## TODO

- [x] 完成NCHW->NHWC模型转换脚本
- [ ] 基于ResNet50用转换后的权重跑通网络，验证正确性
- [ ] 合并PR