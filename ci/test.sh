set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
python3 -m pip install -r test-requirements.txt --user
if [ -f requirements.txt ]; then pip install --user -r requirements.txt; fi
python3 -m pip install oneflow --user -U -f https://staging.oneflow.info/branch/master/cu102
python3 -m pip install oneflow --user gast==0.3.3
python3 setup.py install
# python3 examples/tensorrt_qat/test_lenet_qat_train.py
# python3 -m pytest -s examples/tensorrt_qat/test_lenet_qat.py
python3 -m pytest examples/oneflow2onnx
python3 -m pytest examples/x2oneflow/pytorch2oneflow/nodes
python3 -m pytest examples/x2oneflow/pytorch2oneflow/models
python3 -m pytest examples/x2oneflow/tensorflow2oneflow/nodes
python3 -m pytest examples/x2oneflow/tensorflow2oneflow/models
python3 -m pytest examples/x2oneflow/paddle2oneflow/nodes
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_alexnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_darknet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_densenet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_dpn.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_efficientnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_ghostnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_googlenet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_inceptionv3.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_inceptionv4.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv1.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv2.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_mobilenetv3.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_regnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_repvgg.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_res2net.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_resnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_resnext.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_se_resnext.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_shufflenet_v2.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_squeezenet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_vggnet.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_vision_transformer.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_xception_deeplab.py
python3 -m pytest examples/x2oneflow/paddle2oneflow/models/test_xception.py
