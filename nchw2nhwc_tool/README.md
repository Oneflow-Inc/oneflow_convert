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
python3 inference.py --log_dir="./log" --model_load_dir="./resnet50" --image_path="./fish.jpg" --channel_last=False
```

## TODO

- [ ] 基于ResNet50完成NCHW->NHWC权重转换，并使用转换后的权重跑通网络，验证正确性
- [ ] 重构代码，提供一个更加友好的接口