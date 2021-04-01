#手順

#ファイルのフォルダを選択
#ファイルのリストを作成
#最後のファイルを保存する。

import os
import numpy as np
import cv2
import numpy as np
from tomomi import preprocessing

object_name = 'hazelnut'
#ファイルのフォルダを選択

# data_dir = 'C:\\Users\\seong\\Google ドライブ\\datasets\\mvtec_anomaly_detection\\'\
#            +object_name+'\\train\\good\\'
data_dir ='./mvtec_dB'
#ファイルのリストを作成
print(data_dir)
if os.path.isdir(data_dir):
    file_list = os.listdir(data_dir)
    num_of_files = len(file_list)

else:

    pass

print('file list: ', file_list)
print('num of files: ', num_of_files)


#ファイルのリストの要素を取り出し、(for文)
#numpy hstackで横につなげる (capsule 220,220,220)
colorR = 34
colorG = 34
colorB = 39

img_array = np.zeros((num_of_files,900,900,3),np.uint8)
# img_blank = 255*np.ones((num_of_files,900,900,3),np.uint8) #for bottle
img_blank = np.ones((num_of_files,900,300,3),np.uint8) #for hazelnut
img_blank[:, :, :] = (colorB, colorG, colorR)
img_hori = np.zeros((num_of_files, 900,1200,3),np.uint8)

for i, fName in enumerate(file_list):

    # print(i, fName)
    #make a path + file name
    path_fName = os.path.join(data_dir, fName)
    # print(i, path_fName)

    img = preprocessing.imread(path_fName)
    img = cv2.resize(img,(900,900))

    h, w, c = img.shape #(900,900,3)
    img_array[i] = img
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    img_hori[i] = np.concatenate((img_array[i],img_blank[i]),1)
    print(img_hori[i].shape)
    # cv2.imshow('test',img_hori[i])
    # cv2.waitKey(0)
    img_hori_total = np.hstack(img_hori)




#保存

fName_hori = './image/img_hori_'+object_name+'.png'
preprocessing.imwrite(fName_hori, img_hori_total)

print('Work done!')

