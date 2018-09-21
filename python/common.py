import numpy as np
import cv2
import os
from time import time

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

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

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

def findOpticalFlow(inputVideo, outputVideo, useCuda = False, printFrames = False):
    cap = cv2.VideoCapture(inputVideo)
    # Define the codec and create VideoWriter object
    ret, prev = cap.read()
    # codec = cv2.VideoWriter_fourcc(*'XVID') #cv2.VideoWriter_fourcc('M','J','P','G')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print('framerate: ', fps)
    # out = cv2.VideoWriter(outputVideo, codec, fps, (prev.shape[0], prev.shape[1]))
    g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    count = 1
    start = time()
    while(cap.isOpened()):
        ret, next = cap.read()
        if ret==True:
            g_next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
            if useCuda:
                flow = cv2.pythoncuda.gpuOpticalFlowFarneback(g_prev, g_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                outputFile = 'output/gpu/gpu_frame_{}.png'.format(count)
            else:
                flow = cv2.pythoncuda.cpuOpticalFlowFarneback(g_prev, g_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                outputFile = 'output/cpu/cpu_frame_{}.png'.format(count)

            output = draw_flow(g_prev, flow)
            cv2.imwrite(outputFile, output)
            g_prev = g_next
            if printFrames:
                print('frame: ', count)
            count += 1
        else:
            break
    cap.release()
    count -= 1
    # out.release()
    totaltime = time() - start
    speed = count/totaltime
    if useCuda:
        print('total time in optical flow GPU processing: {:0.4f} sec, for: {} frames. FPS: {:0.2f}'.format(totaltime, count, speed))
    else:
        print('total time in optical flow CPU processing: {:0.4f} sec, for: {} frames. FPS: {:0.2f}'.format(totaltime, count, speed))