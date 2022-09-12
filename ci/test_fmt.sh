set -ex
python3 -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
python3 -m pip install --user --upgrade pip
pip install pylint==2.4.4
pylint --rcfile=tools/pylintrc --ignore=version.py --disable=cyclic-import oneflow_onnx examples/*.py tools -j 0
