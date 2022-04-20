import cv2
import numpy as np

# read image
img = cv2.imread('ball.png')
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
imgSmall = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# hh, ww = img.shape[:2]
# hh2 = hh // 2
# ww2 = ww // 2

# # define circles
# radius = 70
# xc = hh // 2
# yc = ww // 2

# mask = np.zeros_like(img)
# mask = cv2.circle(mask, (xc,yc), radius, (255,255,255), -1)

# # put mask into alpha channel of input
# result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask[:,:,0]

# Bitwise-and for ROI
# ROI = cv2.bitwise_and(img, mask)

# # Crop mask and turn background white
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# x,y,w,h = cv2.boundingRect(mask)
# result = ROI[y:y+h,x:x+w]
# mask = mask[y:y+h,x:x+w]
# result[mask==2] = (255,255,255)

mask = imgSmall[..., 2] != 0
img[mask] = imgSmall[..., :2][mask]

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()