#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import pickle

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# CLASSES = ('__background__',
        # 'faces', 
        # 'lps')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5, image_name='image'):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
                          # edgecolor='red', linewidth=3.5)
            )
        # ax.text(bbox[0], bbox[1] - 2,
                # '{:s} {:.3f}'.format(class_name, score),
                # bbox=dict(facecolor='blue', alpha=0.5),
                # fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    # __ Saving instead of plotting __ #
    
    img_name = "6_demo_{}_{}.jpg".format(image_name.split('.')[0], class_name)
    print(img_name)
    plt.savefig(img_name, dpi=300)

def demo(sess, net, dir_name, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', 'panos', '2', image_dir_name, im_name
    im_file = os.path.join(dir_name, im_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.7
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH, image_name=image_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('--dir', dest='dir_name', help='Directory containing examples')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    dir_name = args.dir_name
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/train/default/vgg16_faster_rcnn_iter_6600.ckpt'
    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/voc_2007_trainval'
    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.ckpt'

    tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/default/train/default/res101_faster_rcnn_iter_80000.ckpt'
    tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/default/train/default/vgg16_allnet_train_iter_20000.ckpt'
    # tfmodel = 'output/vgg16/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_iter_110000.ckpt'

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    net = vgg16(batch_size=1)
    net.create_architecture(sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])

    # net.create_architecture(sess, "TEST", 3,
                          # tag='default', anchor_scales=[2,4,8,16,32])

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # Original demo images
    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                # '001763.jpg', '004545.jpg']

    # Best resolution 512*3072

    done_file = "done.pkl"

    try :
        with open(done_file, 'rb') as fp:
            done = pickle.load(fp)
    except:
        done = []

    print(dir_name)
    i = 0

    probs_im = [
        # lps
        '2000010003553.jpg',
        '2000010003657.jpg',
        '2000010004195.jpg',
        '2000010004575.jpg',
        '2000010004839.jpg',
        '2000010004854.jpg',
        '2000010004866.jpg',
        '2000010004938.jpg',
        '2000010005015.jpg',
        '2000010005081.jpg',
        '2000010005187.jpg',
        '2000010005240.jpg',
        '2000010005551.jpg',
        '2000010005563.jpg',
        '2000010005582.jpg',
        '2000010010490.jpg',
        '2000010010502.jpg',
        '2000010010618.jpg',
        '2000010002088.jpg',
        '2000010006688.jpg',
        '2000010006969.jpg',
        '2000010006970.jpg',
        '2000010006984.jpg',
        # faces
        '2000010004765.jpg',
        '2000010003360.jpg',
        '2000010003385.jpg',
        '2000010003495.jpg',
        '2000010003517.jpg',
        '2000010003573.jpg',
        '2000010003799.jpg',
        '2000010003954.jpg',
        '2000010003958.jpg',
        '2000010005544.jpg',
        '2000010007065.jpg',
        ]

    probs_im = ['2000010006628.jpg',]
    for im_name in os.listdir(dir_name):
        # if im_name in done:
            # continue

        i += 1
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}{}'.format(dir_name, im_name))
        demo(sess, net, dir_name, im_name)

        # save in pickle
        done.append(im_name)

        if i%20 == 0:

            print("Pickle dump...")
            with open(done_file, 'wb') as fp:
                pickle.dump(done, fp)

    plt.show()
