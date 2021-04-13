set -ex
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install --upgrade pip
python3 -m pip install flake8 pytest
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
python3 -m pip install oneflow --user -f https://staging.oneflow.info/branch/master/cu102
python3 setup.py install
python3 -m pytest examples/oneflow2onnx
python3 -m pytest examples/x2oneflow/pytorch2oneflow/nodes
python3 -m pytest examples/x2oneflow/tensorflow2oneflow/nodes
python3 -m pytest examples/x2oneflow/paddle2oneflow/nodes