#!/usr/bin/env python
import numpy as np
import cv2 as cv

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
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
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv.VideoCapture('video/vtest.avi')
    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    codec = cv.VideoWriter_fourcc('M','J','P','G')
    fps = cam.get(cv.CAP_PROP_FPS)
    print('framerate: ', fps)
    out = cv.VideoWriter('video/cpu_output.avi', codec, fps, (prev.shape[0], prev.shape[1]))
    count = 1
    while True and count < 20:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray
        output = draw_flow(gray, flow)
        out.write(output)
        count += 1
    cam.release()
    out.release()
