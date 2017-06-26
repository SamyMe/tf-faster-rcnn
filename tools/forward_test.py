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
import re

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from eval import annotation, recall_iou


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'lp', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'face', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# CLASSES = ('__background__',
        # 'faces', 
        # 'lps')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def forward(sess, net, im_file, CONF_THRESH=0.8, NMS_THRESH=0.7):
    """ Detect object classes in an image using pre-computed object proposals.
	Returns: Result, time"""

    results = {'face': [],
               'lp': []}

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()

    for cls in ('face', 'lp') :# enumerate(CLASSES[1:]):
        cls_ind = CLASSES.index(cls) # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        results[cls] = [
                [x0, y0, x1-x0, y1-y0] for (x0, y0, x1, y1, score) in dets[keep, :]]

    return results, timer.total_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--img_dir', dest='img_dir', default=None, help='Directory containing examples')
    parser.add_argument('--annotation_dir', dest='anno_dir', default=None, help='Directory containing examples')
    parser.add_argument('--img', dest='img_path', default=None, help='Image to pass forward examples')
    parser.add_argument('--model', dest='tfmodel', default=None, help='Trained model to restore')
    parser.add_argument('--min_recall', dest='min_recall', default=None, help='Minimum recall expected')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    img_dir = args.img_dir
    anno_dir = args.anno_dir
    im_file = args.img_path
    tfmodel = args.tfmodel
    min_recall = args.min_recall

    if min_recall == None:
        min_recall = 0.8

    # Original vgg16 model
    # tfmodel = 'output/vgg16/voc_2007_trainval+voc_2012_trainval/vgg16_faster_rcnn_iter_110000.ckpt'
    # Last layer fine tuned
    # tfmodel = '/hoLast layer fe/blur/Documents/git/tf-faster-rcnn/output/default/train/default/res101_faster_rcnn_iter_80000.ckpt'
    # Whole model fine tuned
    # tfmodel = '/home/blur/Documents/git/tf-faster-rcnn/output/default/train/default/vgg16_allnet_train_iter_100000.ckpt'

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

    # One pass
    if im_file != None:
        print('Forward pass for {}'.format(im_file))
        det = forward(sess, net, im_file)

        # Evaluation
        im_id = re.findall('\d{13}', im_file)[0]
        print(im_id)
        gt = annotation(im_id, anno_dir)
        result = recall_iou(det, gt)

    # Pass over foldeO
    if img_dir != None:
        complete_recall = 0
        i = 0
        for im_name in os.listdir(img_dir):
            i += 1
            # print('Forward pass for {}{}'.format(img_dir, im_name))
            im_file = os.path.join(img_dir, im_name)
            det, det_time = forward(sess, net, im_file)

            # Evaluation
            im_id = re.findall('\d{13}', im_file)[0]
            gt = annotation(im_id, anno_dir)
            result = recall_iou(det, gt)
            print("{}  |  {} sc  |  {}".format(im_id, det_time, result))
            complete_recall += result[0]

	complete_recall /= len(os.listdir(img_dir))
        print("Complete recall for {} images = {}".format(len(os.listdir(img_dir)), complete_recall))
	if complete_recall < min_recall:
		raise ValueError('The recall is not good enough !\nModel doesn\'t seem to have trained well.')
