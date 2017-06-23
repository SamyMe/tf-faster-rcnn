# mbuckler/tf-faster-rcnn-deps
#
# Dockerfile to hold dependencies for the Tensorflow implmentation
# of Faster RCNN

FROM nvidia/cuda:8.0-cudnn5-devel
WORKDIR /root

# Default keyboard conf
COPY ./docker/keyboard /etc/default/keyboard

# Get required packages
RUN apt-get update && \
  apt-get install vim \
                  python-pip \
                  python-dev \
                  python-opencv \
                  python-tk \
                  libjpeg-dev \
                  libfreetype6 \
                  libfreetype6-dev \
                  zlib1g-dev \
                  cmake \
                  wget \
                  cython \
                  git \
                  mlocate \
                  -y
                  
# Get required python modules
RUN pip install --upgrade pip
RUN pip install image \
                scipy \
                matplotlib \
                PyYAML \
                numpy \
                easydict \
                shapely \
                h5py \
                tensorflow-gpu

# Update numpy
RUN pip install -U numpy

# Install nvidia-cuda-toolkit for libcuda.so.1
RUN apt-get update
RUN apt-get install cuda nvidia-cuda-toolkit -y 
RUN apt-get install libgeos-dev -y

# Clone project
RUN mkdir /ai
WORKDIR /ai/tf-faster-rcnn
# Copying script files
ADD ./data  ./data
ADD ./experiments ./experiments
ADD ./lib  ./lib
ADD ./tools ./tools

# Add CUDA to the path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
ENV PYTHONPATH /root/coco/PythonAPI
RUN ldconfig

# Build the Cython modules
WORKDIR /ai/tf-faster-rcnn/lib
RUN make clean
RUN make 

# Copy config.py
WORKDIR /ai/tf-faster-rcnn
COPY  ./docker/config.py lib/model/

# To run the container with bash prompt:
# nvidia-docker run -it -v /path/on/disck/:/data faster-rcnn bash
# To launch train:
# python tools/trainval_net.py --imdb mappy_train --imdbval mappy_test --iters 15 --net vgg16 --weight /data/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_iter_110000.ckpt
