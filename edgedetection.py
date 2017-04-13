import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img_00365.jpg',1)
# img = cv2.imread('cropped_0_img_00072.jpg',0)

print img.shape
img=img[500:, :600  ]

kernel = np.ones((3,3),np.float32)/9
img = cv2.filter2D(img,-1,kernel)

edges = cv2.Canny(img,5,25)
plt.figure()
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# plt.figure()
# # converting to gray scale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray=img
# # remove noise
# img = cv2.GaussianBlur(gray,(3,3),0)

# # convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plt.show()