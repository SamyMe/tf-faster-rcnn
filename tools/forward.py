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
import urllib

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from eval import annotation, recall_iou
import numpy


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'lp', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'face', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}


def forward(sess, net, img, CONF_THRESH=0.8, NMS_THRESH=0.7):
    """ Detect object classes in an image using pre-computed object proposals.
	Returns: Result, time"""

    results = {'face': [],
               'lp': []}

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, img)
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


def iterate_over_query(query):
    for im_id in query:
        pano_hexa = urllib.urlopen(
                "http://{}:{}@{}/bo/images/fullRaw/tile/{}".format(
                    cfg.BO.id, cfg.BO.password, 
		    cfg.BO.address_port,
		    construct_path(im_id))
                ).read()

        pano = cv2.imdecode(numpy.fromstring(pano_hexa, numpy.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        yield im_id, pano 


def push_detection(det):
    # Push detections through the API 
    for key, val in det.items():
        print(key, len(val))


def construct_path(id_val):
       """Constructing path to a pano from its _id
       """
       id_val = str(id_val)
       path = id_val[:3] + "/" + id_val[3:6] + "/" + id_val[6:9] + "/"
       path += id_val
       return path


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='net_name', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--model', dest='tfmodel', default=None, help='Trained model to restore')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    net_name = args.net_name
    tfmodel = args.tfmodel

    min_recall = 0.5

    if min_recall == None:
        min_recall = 0.8

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if net_name == 'vgg16':
        net = vgg16(batch_size=1)
    elif net_name == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError

    net = vgg16(batch_size=1)
    net.create_architecture(sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # Get images to annotate
    query = [1000042997570, 2000010009999]
    image_set = iterate_over_query(query)
    if image_set != None:
        i = 0
        for im_id, img in image_set:
            i += 1
            det, det_time = forward(sess, net, img)
            print("n {} : {}  |  {} sc ".format(i, im_id, det_time))

            # Push to the API
            push_detection(det)
