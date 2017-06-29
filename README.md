# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).



### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.0** format. If you are using an older version (r0.1-r0.12), please check out the v0.12 release. While it is not required, for experimenting the original RoI pooling (which requires modification of the C++ code in tensorflow), you can check out my tensorflow [fork](https://github.com/endernewton/tensorflow) and look for ``tf.image.roi_pooling``.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. I use 1.6.
  - Docker users: Since recent upgrade Tensorflow **r1.0**, the docker image on docker hub (https://hub.docker.com/r/mbuckler/tf-faster-rcnn-deps/) is no longer valid. However, you can still build your own image by using dockerfile located at `docker` folder (cuda 8 version, as it is required by Tensorflow **r1.0**.) And make sure following Tensorflow installation to install and use nvidia-docker[https://github.com/NVIDIA/nvidia-docker]. Last, after launching the container, you have to build the Cython modules within the running container. 

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/endernewton/tf-faster-rcnn.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal)  | sm_52  |
  | Grid K520 (AWS g2.2xlarge)  | sm_30  |
  | Tesla K80 (AWS p2.xlarge)   | sm_37  |


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```


### Setup Training data

Training data should be included in a directory following this structure:

data_dir/
---------- data/
---------- data/Images/ (id.jpg)
---------- data/Annotations/ (id.txt)
---------- data/ImageSets/ (train.txt, test.txt)

Where the *Images/* folder contains *jpg* images.
The *Annotations/* folder contains a *txt* annotation file for each images with (x, y, w, h, cls).
And *ImageSets/* contains training / tests set (list of images to use).

### Train model on panos data

1. Download pre-trained model
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
    **Note**: if you cannot download the models through the link, or you want to try more models, you can check out the following solutions and optionally update the downloading script:
- Another server [here](http://gs11655.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/).
- Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. Edit lib/model/config.py . Set :
__C.DATA_FACTORY = '/path/to/data_dir/'
__C.DATA_IMDB = '/path/to/data_dir/data/'



3. Launch training 

```bash
python tools/trainval_net.py \
	--imdb mappy_train --imdbval mappy_test \
	--iters #nb_of_iteration \
	--net vgg16 \
	--weight /path/to/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_iter_110000.ckpt 
```

### Demo trained models

The demo script forwards a bunch of images through the trained model, plots the results, and save them to an output directory.

```bash
python tools/demo.py \ 
	--img_dir /path/to/image/dir/  \
	--output_dir /path/to/output/dir/ \
	--model /path/to/model_iter_.ckpt
```

### Test model :

The test script takes as input a *trained model*, a *set of images* and their *ground truth annotations* (gt). It computes the score the model does on the data with a recall over the *gt*. i.e. the precentage of object mentioned in the annotatiosn files did the model get.


```bash
python tools/forward.py \
	--img_dir /path/to/image/dir/ \
	--annotation_dir /path/to/gt/annotations/ \
	--model /path/to/model_iter_.ckpt
```

### Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{chen17implementation,
        Author = {Xinlei Chen and Abhinav Gupta},
        Title = {An Implementation of Faster RCNN with Study for Region Sampling},
        Journal = {arXiv preprint arXiv:1702.02138},
        Year = {2017}
    }

For convenience, here is the faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
