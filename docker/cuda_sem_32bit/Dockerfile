# Use the base cudolfinx image 
FROM benpachev/cudolfinx:v0.10.0-cuda12.6

WORKDIR /root

# Various package versions for SEM use case
ARG TORCH_VERSION=2.10.0
ARG TV_VERSION=0.25.0
ARG TORCH_CUDA=https://download.pytorch.org/whl/cu128

ARG MDANALYSIS_VERSION=2.10.0

ARG PANDAS_VERSION=3.0.1

RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TV_VERSION} --index-url ${TORCH_CUDA} && \
    pip install --no-cache-dir mdanalysis==${MDANALYSIS_VERSION} && \
    pip install --no-cache-dir pandas==${PANDAS_VERSION} && \
    pip install --no-cache-dir jupyter #Too lazy to figure out version :p

#This doesn't actually work at the moment (You need to set export these variables in the running container atm :p)
ENV CC=gcc
ENV CXX=g++
