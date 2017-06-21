# mbuckler/tf-faster-rcnn-deps
#
# Dockerfile to hold dependencies for the Tensorflow implmentation
# of Faster RCNN

FROM nvidia/cuda:8.0-cudnn5-devel
WORKDIR /root

# Default keyboard conf
COPY ./keyboard /etc/default/keyboard

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
WORKDIR /ai
RUN git clone https://github.com/SamyMe/tf-faster-rcnn

# Add CUDA to the path
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV CUDA_HOME /usr/local/cuda
ENV PYTHONPATH /root/coco/PythonAPI
RUN ldconfig

# Build the Cython modules
WORKDIR /ai/tf-faster-rcnn/lib
RUN make clean
RUN make 

# Copy 1388 Images and config.py
WORKDIR /ai/tf-faster-rcnn
RUN mkdir -p /home/blur/Documents/data/_output_imdb
COPY ./data/nk/data  /home/blur/Documents/data/_output_imdb/data
COPY  config.py lib/model/

# Download vgg16 pretrained on imagenet
RUN mkdir -p data/imagenet_weights
WORKDIR data/imagenet_weights
RUN wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
RUN tar -xzvf vgg_16_2016_08_28.tar.gz
RUN mv vgg_16.ckpt vgg16.ckpt

WORKDIR /ai/tf-faster-rcnn
# RUN python tools/forward.py --img data/demo/000456.jpg 
RUN python tools/trainval_net.py --imdb mappy_train --imdbval mappy_test --iters 15 --net vgg16 --weight data/imagenet_weights/vgg16.ckpt
