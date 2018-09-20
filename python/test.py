import cv2
import numpy as np
import common as cm
from moviepy.editor import VideoFileClip


g_prev = None
def process_image(img):
    g_next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if g_prev is None:
        g_prev = g_next
    else:
        g_next = img
    flow = cv2.pythoncuda.opticalFlowFarneback(g_prev, g_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print('frame processed..')
    return cm.draw_flow(g_prev, flow)
    
def findOpticalFlow(inputVideo, outputVideo, useCuda = False):
    video = VideoFileClip(inputVideo).subclip(1,3)
    processed_video = video.fl_image(process_image)
    processed_video.write_videofile(outputVideo, audio=False)
    processed_video.reader.close()
    processed_video.audio.reader.close_proc()
    
findOpticalFlow('video/vtest.avi', 'video/cpu_output.avi')