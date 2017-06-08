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
    
    img_name = "demo_{}_{}.jpg".format(image_name.split('.')[0], class_name)
    plt.savefig(img_name, dpi=300)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', 'panos', '2', image_name)
    im_file = os.path.join(cfg.DATA_DIR, 'demo', 'panos', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.9
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
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/train/default/vgg16_faster_rcnn_iter_6600.ckpt'
    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/voc_2007_trainval'
    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/vgg16/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.ckpt'

    tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/default/train/default/res101_faster_rcnn_iter_600.ckpt'
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

    # 200000 images
    im_names = [
            '2000010000005.jpg', '2000010000007.jpg', '2000010000018.jpg', '2000010000023.jpg', '2000010000025.jpg', '2000010000027.jpg', '2000010000029.jpg', '2000010000031.jpg', '2000010000033.jpg', '2000010000006.jpg', '2000010000008.jpg', '2000010000022.jpg', '2000010000024.jpg', '2000010000026.jpg', '2000010000028.jpg', '2000010000030.jpg', '2000010000032.jpg', '2000010000034.jpg'
            ]

    im_names = [
'2000010000008.jpg', '2000010000022.jpg',
'1000038095323.jpg', '1000041373831.jpg', '1000041375422.jpg', '1000041375985.jpg', '1000041383807.jpg', '1000041389784.jpg', '1000041390341.jpg', '1000041390379.jpg', '1000041391977.jpg', '1000041373341.jpg', '1000041374057.jpg', '1000041375423.jpg', '1000041375986.jpg', '1000041383818.jpg', '1000041389844.jpg', '1000041390343.jpg', '1000041390406.jpg', '2000010000008.jpg', '1000041373342.jpg', '1000041374067.jpg', '1000041375568.jpg', '1000041377051.jpg', '1000041386232.jpg', '1000041389847.jpg', '1000041390344.jpg', '1000041390773.jpg', '2000010000022.jpg', '1000041373343.jpg', '1000041374285.jpg', '1000041375972.jpg', '1000041377077.jpg', '1000041388478.jpg', '1000041389982.jpg', '1000041390345.jpg', '1000041391974.jpg', '1000041373741.jpg', '1000041374286.jpg', '1000041375977.jpg', '1000041383563.jpg', '1000041389284.jpg', '1000041390226.jpg', '1000041390372.jpg', '1000041391975.jpg', '1000041373802.jpg', '1000041374288.jpg', '1000041375984.jpg', '1000041383564.jpg', '1000041389545.jpg', '1000041390339.jpg', '1000041390373.jpg', '1000041391976.jpg'
              ]

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/panos/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
