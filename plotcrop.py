import cv2
import json
f=open('boundingbox_YFT.json')
data=json.load(f)
print data
ss=data['img_07569.jpg']
img=cv2.imread('img_07569.jpg',1)
cv2.namedWindow('image')
cv2.imshow('image',img)