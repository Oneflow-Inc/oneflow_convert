set -ex
python -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python -m pip install --user --upgrade pip
python -m pip install --user flake8 pytest
python -m pip install nvidia-pyindex
python -m pip install nvidia-tensorrt==8.0.0.3
python -m pip install pycuda
if [ -f requirements.txt ]; then pip install --user -r requirements.txt; fi
python -m pip install oneflow --user -U -f https://staging.oneflow.info/branch/master/cu110
python setup.py install
python examples/tensorrt_qat/test_lenet_qat_train.py
python -m pytest -s examples/tensorrt_qat/test_lenet_qat.py
python examples/tensorrt_qat/test_mobilenet_qat_train.py
python -m pytest -s examples/tensorrt_qat/test_mobilenet_qat.py
python -m pytest examples/oneflow2onnx
python -m pytest examples/x2oneflow/pytorch2oneflow/nodes
python -m pytest examples/x2oneflow/pytorch2oneflow/models
python -m pytest examples/x2oneflow/tensorflow2oneflow/nodes
python -m pytest examples/x2oneflow/tensorflow2oneflow/models
python -m pytest examples/x2oneflow/paddle2oneflow/nodes
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_alexnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_darknet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_densenet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_dpn.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_efficientnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_ghostnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_googlenet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_inceptionv3.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_inceptionv4.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv1.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv2.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv3.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_regnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_repvgg.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_res2net.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_resnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_resnext.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_se_resnext.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_shufflenet_v2.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_squeezenet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_vggnet.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_vision_transformer.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_xception_deeplab.py
python -m pytest examples/x2oneflow/paddle2oneflow/models/test_xception.py
