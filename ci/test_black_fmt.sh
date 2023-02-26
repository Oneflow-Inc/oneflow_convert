set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
python3 -m pip install click==8.0.0
python -m pip install black==22.3.0

black -l 200 --check oneflow_onnx/ examples/
