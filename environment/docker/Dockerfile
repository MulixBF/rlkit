# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        git \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        software-properties-common \
        net-tools \
        unzip \
        vim \
        virtualenv \
        wget \
        xpra \
        xserver-xorg-dev \
        swig \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6
RUN virtualenv --python=python3.6 /opt/env

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200

COPY ./mjkey.txt /root/.mujoco/

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH="/opt/env/bin:/root/.mujoco/mujoco200/bin:$PATH"

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && apt-get install -y libav-tools

# For some reason this works despite an error showing up...
RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384

COPY requirements.txt /requirements.txt
RUN python -m pip install "numpy==1.11.3" Cython
RUN python -m pip install -r requirements.txt

RUN mkdir /root/code
WORKDIR /root/code

