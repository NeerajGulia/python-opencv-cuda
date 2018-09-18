#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    outputdir = 'output/'
    inputdir = 'images/'
    prev = cv2.imread(os.path.join(inputdir, 'frame0.png'), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('prev', prev)
    current = cv2.imread(os.path.join(inputdir, 'frame1.png'), cv2.IMREAD_GRAYSCALE)
    # input('press any key to continue...')
    flow = cv2.pythoncuda.gpuOpticalFlowFarneback(prev, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow = cv2.calcOpticalFlowFarneback(prev, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    cv2.imwrite(os.path.join(outputdir, 'cpu_flow.png'), draw_flow(prev, flow))
    print('image saved...')
