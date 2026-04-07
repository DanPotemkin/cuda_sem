# Dockerfile describing end-user CUDA-accelerated FEniCS environments
# Modified version of the DOLFINx end user Docker file
#
# Authors:
# Benjamin Pachev <benjamin.pachev@gmail.com>
#

ARG PYVISTA_VERSION=0.44.2

# Used to set the correct PYTHONPATH for the real and complex install of
# DOLFINx
ARG PYTHON_VERSION=3.12
# Base image for end-user images
ARG BASEIMAGE=dolfinx-64bit
ARG CUDOLFINX_CMAKE_BUILD_TYPE="Release"

FROM ${BASEIMAGE} as cudolfinx
LABEL description="cuDOLFINx"

ARG PYTHON_VERSION

WORKDIR /src
RUN git clone https://github.com/DanPotemkin/cuda-dolfinx.git
 
RUN cd cuda-dolfinx && \
    mkdir -p build-real && \
    cd build-real && \
    PETSC_ARCH=linux-gnu-real64-64-cuda cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=${CUDOLFINX_CMAKE_BUILD_TYPE} -DMPI_C_COMPILER=/usr/local/bin/mpicc -DMPI_CXX_COMPILER=/usr/local/bin/mpicxx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DHDF5_ENABLE_PARALLEL=ON ../cpp && \
    ninja install && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-real64-64-cuda CMAKE_ARGS="-DMPI_C_COMPILER=/usr/local/bin/mpicc -DMPI_CXX_COMPILER=/usr/local/bin/mpicxx -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DHDF5_ENABLE_PARALLEL=ON" pip -v install \
      --config-settings=cmake.build-type="${CUDOLFINX_CMAKE_BUILD_TYPE}" --config-settings=install.strip=false --no-build-isolation --check-build-dependencies \
      --target /usr/local/dolfinx-real/lib/python${PYTHON_VERSION}/dist-packages --no-dependencies --no-cache-dir '.'

ARG PYVISTA_VERSION

WORKDIR /root

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pyvista from PyPI. pyvisa depends on (py)vtk), but vtk wheels are not
# available on pypi for linux/arm64, so we use a custom build wheel.
# matplotlib improves plotting quality with better color maps and
# properly rendering colorbars.
# trame is the preferred backend for pyvista.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    pip install matplotlib; \
    case "$dpkgArch" in amd64) \
      pip install --no-cache-dir pyvista[trame]==${PYVISTA_VERSION} ;; \
    esac; \
    case "$dpkgArch" in arm64) \
      pip install --no-cache-dir https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.3.0-cp312/vtk-9.3.0.dev0-cp312-cp312-linux_aarch64.whl && \
      pip install --no-cache-dir pyvista[trame]==${PYVISTA_VERSION} ;; \
    esac; \
    pip cache purge

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
