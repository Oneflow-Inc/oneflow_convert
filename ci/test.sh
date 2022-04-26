set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt --user; fi
python3 -m pip install -f https://release.oneflow.info oneflow==0.7.0+cu112
python3 setup.py install
pip install flowvision==0.0.3 --user
python3 -m pytest examples/oneflow2onnx/models/CPU
python3 -m pytest examples/oneflow2onnx/models/GPU
python3 -m pytest examples/oneflow2onnx/nodes/CPU
python3 -m pytest examples/oneflow2onnx/nodes/GPU
