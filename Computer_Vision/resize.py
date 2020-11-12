import sys
import cv2
import numpy as np

img1 = cv2.imread('maxresdefault.jpg', cv2.IMREAD_COLOR)

ww = 500
hh = 400

if img1 is None:
    print('Image load failed!')
    sys.exit()

# 영상의 속성 참조
print('type(img1):', type(img1))
print('img1.shape:', img1.shape)
print('img1.dtype:', img1.dtype)

# 영상의 크기 참조
h, w = img1.shape[:2]
print('img1 size: {} x {}'.format(w, h))

if len(img1.shape) == 2:
    print('img1 is a grayscale image')
elif len(img1.shape) == 3:
    print('img1 is a truecolor image')

cv2.imshow('img1', img1)
cv2.waitKey()

srcQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32) #좌상단, 우상단, 우하단, 좌하단
dstQuad = np.array([[0, 0], [ww-1, 0], [ww-1, hh-1], [0, hh-1]], np.float32) #좌상단, 우상단, 우하단, 좌하단

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad) #3x3 투시변환 행렬
dst = cv2.warpPerspective(img1, pers, (ww, hh))

black = np.zeros((hh+100, ww+100, 3), np.uint8)


for www in range(ww):
    for hhh in range(hh):
        for color in range(3):
            black[50+hhh, 50+www, color] = dst[hhh, www, color]

cv2.imshow('dst', dst)
cv2.waitKey()

cv2.imshow('black', black)
cv2.waitKey()

cv2.imwrite('resized.jpg', black)

cv2.destroyAllWindows()