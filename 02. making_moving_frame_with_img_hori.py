import os
import numpy as np
import cv2
import numpy as np
from tomomi import preprocessing

object_name = 'hazelnut'

img_fName = './image/img_hori_'+object_name+'.png'

img_source = preprocessing.imread(img_fName)
h,w,c = img_source.shape

print('height',h)
print('total_width',w)
print('channels',c)
width_origin = 1200

window_h = h
window_w = int(1.2*width_origin)
size = (window_w, window_h) #w,hの順に注意

vel = 8  #[pixel/frame] 1
frame_count = (w-window_w)//vel
print('frame count:',frame_count)
# frame_count = 330
frame_rate = 30 #FPS 300
codec = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('./image/test-'+object_name+'.mp4', codec, frame_rate, size)


for i in range(frame_count):
    #Crop
    #(x,y), w,h
    #img[y:y+h, x:x+w]

    img_frame = img_source[0:h, int(vel*i) : int(vel*i) + window_w]
    # print(i, vel*i)
    # cv2.imshow('test', img_frame)
    # cv2.waitKey(0)
    video.write(img_frame.astype(np.uint8))


video.release()

