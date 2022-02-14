---
layout: post
title: Building an NVIDIA Docker Container with Podman
author: Artan Zandian
date: Feb 05, 2022
excerpt: "The base image for Jetson Nano comes with PyTorch preinstalled. In this blog I will go through creation of a docker image to run a Keras model on Jetson Nano."
---
The base image for Jetson Nano comes with PyTorch preinstalled with Jupyter Lab. In this blog I will go through creation of a very light docker image without Jupyter dependencies to run a Keras model on Jetson Nano. I am building this docker image to enable reproducibility for my [Neural Style Transfer](https://artanzand.github.io//neural-style-transfer/) project and to further allow me to run my Keras model on Jetson Nano in Nvidia Docker container (see project [repository](https://github.com/artanzand/neural_style_transfer)). This post assumes reader's basic familiarity with containers and Docker images, and therefore, I will not be justifying my choice of going with a Docker image for this project in detail.
<br>

## Problem Statement

My problem at a high level was that I needed extra packages built on the base Nvidia image, and since it was very likely for me to format my Jetson Nano microSD card due to memory size limitations, I decided to go with a container solution to automate the installation of possible future images.

Further to that I wanted a light image and my only interaction with the image was through a single line of code. Therefore, the image only needed to contain Python, Tensorflow, Keras and related project packages. All available Nvidia packages include Jupyter and unnecessary packages that would make the installation on a 2GB RAM very long.

Here is a brief overview of the process.

- Installing latest JetPack on Jetson Nano.  
- Cross-compiling Docker build setup on an X64 machine.  
- Building a Jetson Nano docker.  
<br>

# Installing JetPack on Jetson Nano

NVIDIA JetPack is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK. JetPack 4.6 is the latest production release, and supports all Jetson modules. Top included features are OpenCV (for computer vision), CUDA, cuDNN, TensorRT.  
For a guided instruction to write the image on your microSD card, and to find the latest image for Jetpack please refer to the [Nvidia Developer](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit#write) webpage. I have also posted and you may find my preferred setup for Jetson Nano in this [blog](https://artanzand.github.io//Setup-Jetson-Nano/).

<br>

# Cross-compiling Docker build setup

The biggest constraint when building a docker image using Jetson Nano Developer kit is the memory size. In my case, I only had 2GB of RAM at my disposal and I was getting multiple errors for lack of memory. Therefore, although the Nvidia Docker is pre-installed on the base OS, cross-compiling Docker on an X64 based machine can save a significant amount of building time considering larger processing power.

This is the birds-eye view of what we are trying to do. We will create a virtual machine to emulate the OS on Jetson Nano (with an aarch64 CPU architecture), will install an Nvidia compatible image builder (podman), and will then create our docker file and push the image to Docker Hub. The image can then be pulled from Jetson Nano.

The first step is to install Docker on your machine (laptop) following official [Docker](https://docs.docker.com/engine/install/ubuntu/) instructions. Then install QEMU which will emulate Jetson Nano CPU architecture on your machine when building Docker containers.  

Based on your operating system you can install QEMU with the instructions provided [here](https://www.qemu.org/download/). QEMU, short for Quick Emulator, is a free open-source virtual machine manager that can execute hardware virtualization. With the help of KVM (Kernel-based Virtual Machine), QEMU can offer fast running speed. Therefore, it develops fast and is estimated to replace VirtualBox and VMware on Linux. However, on Windows, the advantages of QEMU are not significant, because the KVM technology is not applicable on the Windows host machine. But QEMU develops fast, and many people still want to use this VM software on Windows. You may find this [post](https://www.minitool.com/partition-disk/qemu-for-windows.html) for installing QEMU on Windows very useful.

The last step is to install [podman](https://podman.io/). Instead of the default Docker container which has some compatibility issues with Nvidia base image we will be using podman to do the heavy lifting and building the image. We will then push the built image to Docker Hub.  

See the instructions for installing podman [here](https://podman.io/getting-started/installation).

<br>

# Building a Jetson Nano docker

 The most important building block of constructing a docker image is finding the most appropriate base image for our new image. Although newer versions of Nvidia base image are available, I found nvcr.io/nvidia/l4t-base:r32.2 image most compatible with the rest of my project packages. Here is the Dockerfile.

```console
FROM nvcr.io/nvidia/l4t-base:r32.2

# Install python packages
WORKDIR /
RUN apt update && apt install -y --fix-missing \
    make \
    g++ \
    python3-pip \
    libhdf5-serial-dev \
    hdf5-tools
RUN apt update && apt install -y python3-h5py

# Install tensor related packages
RUN pip3 install --upgrade setuptools
RUN pip3 install cython pillow docopt
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install --upgrade numpy
```

You can now pull the base image from nvcr.io, build the image using podman and push to Docker Hub. You will need to change the name of the image. You will be required to login to docker on command line before you can push to the hub.

```console
$podman pull nvcr.io/nvidia/l4t-base:r32.2
$podman build --tag docker.io/artanzandian/keras:0.1.0 . -f ./Dockerfile
$podman push docker.io/artanzandian/keras:0.1.0
```

<br>

# Conclusion

Although this post shows the basic steps of creating an Nvidia Docker container to run a Keras model on Jetson Nano, the reader is highly advised to check for package dependencies when specifying the base image. For example, although TensorFlow 2.0 is available for installation on the Nano it is not recommended due to incompatibilities with the version of TensorRT that comes with the Jetson Nano base OS.

Do you have access to Jetson Nano and want to try out my docker image? Check out my project [repository](https://github.com/artanzand/neural_style_transfer) for updated Dockerfile where you as a reward can also build a stylized version of your image based on your painting of choice.

<br>

## References

- [Nvidia Forums](https://forums.developer.nvidia.com/categories)
- QEMU Installation [guide](https://www.qemu.org/download/)
- How to run Keras model on Jetson - [blog](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/)
