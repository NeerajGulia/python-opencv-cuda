import numpy as np
import cv2
from common import draw_str
from time import time
from sys import maxsize

iterations = 10
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src, runat = 'cpu', limit = 0):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.runat = runat        
        if limit == 0:
            limit = maxsize
        self.frames_limit = limit

    def run(self):
        start = time()
        while True and self.frame_idx < self.frames_limit:
            _ret, frame = self.cam.read()
            if _ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    if self.runat == 'gpu':
                        p1, _st, _err = cv2.pythoncuda.gpuOpticalFlowPyrLK(img0, img1, p0, lk_params["winSize"], lk_params["maxLevel"], iterations)
                        p0r, _st, _err = cv2.pythoncuda.gpuOpticalFlowPyrLK(img1, img0, p1, lk_params["winSize"], lk_params["maxLevel"], iterations)             
                    else:
                        p1, _st, _err = cv2.pythoncuda.cpuOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                        p0r, _st, _err = cv2.pythoncuda.cpuOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray = frame_gray
                if self.runat == 'gpu':
                    outputFile = 'output/gpu/gpu_frame_{}.png'.format(self.frame_idx)
                else:
                    outputFile = 'output/cpu/cpu_frame_{}.png'.format(self.frame_idx)
                cv2.imwrite(outputFile, vis)
                print('frame: ', self.frame_idx)
            else:
                break
        
        totaltime = time() - start
        speed = self.frame_idx/totaltime

        if self.runat == 'gpu':
            print('total time in optical flow GPU processing: {:0.4f} sec, for: {} frames. FPS: {:0.2f}'.format(totaltime, self.frame_idx, speed))
        else:
            print('total time in optical flow CPU processing: {:0.4f} sec, for: {} frames. FPS: {:0.2f}'.format(totaltime, self.frame_idx, speed))    
            
def main():
    import sys
    exit = False
    try:
        runat = sys.argv[1].lower()
        if runat != 'gpu' and runat != 'cpu':
            print('Please mention "gpu" or "cpu"')
            exit = True
    except:
        runat = 'cpu'
    try:
        limit = int(sys.argv[2])
    except:    
        limit = 0
    if not exit:
        print('run at: {}, limit: {}'.format(runat, limit))
        App('video/vtest.avi', runat, limit).run()

if __name__ == '__main__':
    main()
