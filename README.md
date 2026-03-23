Dependencies:
    Docker
    Nvidia-container toolkit (can check by running nvidia-smi in container)

How to setup:

```
git clone git@github.com:DanPotemkin/cuda_sem.git
cd docker
sudo docker build -t cudolfinx .
```

How to run:
```
sudo docker run --gpus all -it cudolfinx

#My nvidia container toolkit defaults to using nvcc for dolfinx which causes errors
export CC=gcc
export CXX=g++
```



