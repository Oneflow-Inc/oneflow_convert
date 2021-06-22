set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
python3 -m pip install --user flake8 pytest
python3 -m pip install nvidia-pyindex
python3 -m pip install nvidia-tensorrt==7.2.3.4
python3 -m pip install pycuda
if [ -f requirements.txt ]; then pip install --user -r requirements.txt; fi
python3 -m pip install oneflow --user -U -f https://staging.oneflow.info/branch/master/cu102
python3 -m pip install --user gast==0.3.3
python3 setup.py install
# python3 examples/tensorrt_qat/test_lenet_qat_train.py
# python3 -m pytest -s examples/tensorrt_qat/test_lenet_qat.py
python3 -m pytest examples/oneflow2onnx
./test_code_gen.sh

