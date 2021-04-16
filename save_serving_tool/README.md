# save_serving_tool

这个工具以ResNet50为例子，介绍如何将OneFlow训练好的模型保存为OneFlow Serving端的模型。

- resnet50.py 是ResNet50的OneFlow模型构建代码。
- save_model.py 是加载ResNet50并转换为Serving端模型的具体代码，如果要转换其它模型，对应替换模型名字即可。
- save_model.sh 将ResNet50模型转换的相关命令写入了这个Bash脚本，用户可对应修改相应参数即可快速完成Serving端的模型保存。

如果用户要转换其它自定义模型，按照这个示例对应修改即可。
