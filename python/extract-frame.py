import cv2
import os

vidcap = cv2.VideoCapture('video/vtest.avi')
success,image = vidcap.read()
count = 1 # start from the 1 instead of 0
directoryname = 'images/'
while success and count < 10:
  cv2.imwrite(os.path.join(directoryname, "frame%d.png" % count), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  
print(count-1, + ' frames extracted successfully')
