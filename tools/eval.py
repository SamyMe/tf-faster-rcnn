from random import random
import numpy as np
import os
import re
from shapely.geometry import box

import _init_paths
from model.config import cfg


def recall_iou(det, gt, threshold=0.01):
    cls_name = {7: 'face', 15: 'lp'}
    iou = []
    for gt_bbox in gt:
        found = False
        best_iou = 0
        x1, y1, x2, y2, cls = gt_bbox
        gt_box = box(x1, y1, x2, y2)
        gt_area = gt_box.area
        if gt_area == 0:
            continue

        for bbox in det[cls_name[cls]]:
            x1, y1, x2, y2 = bbox
            # box(minx, miny, maxx, maxy, ccw=True)
            det_box = box(x1, y1, x2, y2)

            u = det_box.union(gt_box).area
            i = det_box.intersection(gt_box).area

            if i/ gt_area > threshold:
                found = True
                if best_iou < i/gt_area :
                    best_iou = i/gt_area
            
        if found == True:
            iou.append(best_iou)

    if len(gt)!=0:
        recall = float(len(iou))/len(gt)
    else:
        recall = 1

    return recall, iou


def annotation(im_id, annotation_dir):
    # Load Annotation from .txt
    # Returns gt
    
    filename = os.path.join(annotation_dir, im_id + '.txt')
    # print 'Loading: {}'.format(filename)
    with open(filename) as f:
        data = f.read()

    objs = re.findall('\(\d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d+(?:\.\d+)?, \d\)', data)
    num_objs = len(objs)

    gt = []
    
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        coor = re.findall('\d+(?:\.\d+)?', obj)
        x1 = float(coor[0])
        y1 = float(coor[1])
        w = float(coor[2])
        h = float(coor[3])
        cls = int(coor[4]) 

        x2 = x1+w 
        y2 = y1+h 
        
        if cls == 1 :
            cls = 15
        elif cls == 2 :
            cls = 7

        gt.append([x1, y1, x2, y2, cls])

    return gt

