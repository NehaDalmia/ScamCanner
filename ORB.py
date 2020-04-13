import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('IMG_3477.jpg',0)
#cv2.imshow('test',img)
#cv2.waitKey(-1)
orb = cv2.ORB_create(nfeatures=2000)
kp ,des = orb.detectAndCompute(img,None)
print("# kps: {}, descriptors: {}".format(len(kp), des.shape))
img = cv2.drawKeypoints(img,kp,None)
print(des)
cv2.imshow("test",img)
cv2.waitKey(0)
#plt.imshow(img2),plt.show()
