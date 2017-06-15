from random import random

import numpy as np
from shapely.geometry import box


def recall_iou(det, gt, threshold=0.01):
    cls_name = {7: 'face', 15: 'lp'}
    iou = []
    for gt_bbox in gt:
        found = False
        best_iou = 0
        x1, y1, x2, y2, cls = gt_bbox
        gt_box = box(x1, y1, x2, y2)

        for bbox in det[cls_name[cls]]:
            x1, y1, x2, y2 = bbox
            # box(minx, miny, maxx, maxy, ccw=True)
            det_box = box(x1, y1, x2, y2)

            u = det_box.union(gt_box).area
            i = det_box.intersection(gt_box).area

            if i/u > threshold:
                found = True
                if best_iou < i/u :
                    best_iou = i/u
            
        if found == True:
            iou.append(best_iou)

    recall = float(len(iou))/len(gt)

    return recall, iou


if __name__ == "__main__":

    gt = [(1,1,3,3,  7), (1,5,2,6,  7), (5,1,7,3,  7)]
    det = {'face': [[2,1,3,4], [5,1,7,3]]}

    result = recall_iou(det, gt)
    # (0.6666666666666666, [0.4, 1.0])
