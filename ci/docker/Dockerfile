FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="/miniconda3/bin/python -m pip --no-cache-dir install --upgrade" && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && \
    wget https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    /miniconda3/bin/python -m pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple && \
    ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /miniconda3/ -follow -type f -name '*.a' -delete && \
    find /miniconda3/ -follow -type f -name '*.js.map' -delete && \
    /miniconda3/bin/conda clean -afy && \
    rm -f ./Miniconda3-latest-Linux-x86_64.sh
