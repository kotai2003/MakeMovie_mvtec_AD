#https://stackoverflow.com/questions/56787999/python-opencv-realtime-get-rgb-values-when-mouse-is-clicked

import cv2
import numpy as np
from tomomi import preprocessing


def mouseRGB(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #check mouse button down condition
        colorB = image[y,x,0]
        colorG = image[y,x,1]
        colorR = image[y,x,2]
        color = image[y, x]
        print("Red: ", colorR)
        print("Green: ", colorG)
        print("Blue: ", colorB)
        print("BRG Format: ", color)
        print("Coordinates of pixel: X: ", x, "Y: ", y)

        #Generate new blank image
        blank_img = np.zeros_like(img, dtype = np.uint8)
        blank_img[:, :, :] = (colorB, colorG, colorR)
        text = 'R: '+str(colorR) + ' G: '+str(colorG) + ' B: '+str(colorB)
        cv2.putText(blank_img, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (128,128,128),3,cv2.LINE_AA)
        cv2.imshow('blank image', blank_img)

        cv2.waitKey(0)



cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB', mouseRGB)

object_name = 'metal_nut'
#
# img_fName = './image/img_hori_'+object_name+'.png'

image_fName = './mvtec_dB/000.png'
# image_fName = 'C:\\Users\\seonghun\\Google ドライブ\\datasets\\mvtec_anomaly_detection\\'+object_name+'\\train\\good\\000.png'

# img = cv2.imread(image_fName)
img = preprocessing.imread(image_fName)
image = img.copy()

cv2.imshow('mouseRGB',img)
cv2.waitKey(0)













