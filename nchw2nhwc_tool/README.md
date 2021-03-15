### nchw2nhwc_tool

#### 依赖
- oneflow

#### 执行指令

```
python3 inference.py --log_dir="./log" --model_load_dir="./resnet_v15_of_best_model_val_top1_77318" --image_path="./fish.jpg" --channel_last=False
```

## TODO

- [ ] 基于ResNet50完成NCHW->NHWC权重转换，并使用转换后的权重跑通网络，验证正确性
- [ ] 重构代码，提供一个更加友好的接口