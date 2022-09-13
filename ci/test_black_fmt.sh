set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
python3 -m pip install click==8.0.4
python -m pip install black==21.4b2

black -l 200 --check oneflow_onnx/ examples/
