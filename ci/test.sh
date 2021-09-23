set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
python3 -m pip install -r test-requirements.txt --user --extra-index-url https://pypi.ngc.nvidia.com
if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt --user; fi
python3 -m pip install oneflow --user -U -f https://staging.oneflow.info/branch/master/cu110
python3 setup.py install
python3 -m pytest examples/oneflow2onnx
