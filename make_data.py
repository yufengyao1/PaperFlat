import cv2
import random
import numpy as np

img = cv2.imread('test.jpg')
w, h = img.shape[1], img.shape[0]
scale = random.uniform(0.5, 0.9)
scale = scale*816/h
img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
w, h = img.shape[1], img.shape[0]

offset_x1 = random.uniform(-(612-w)/2, (612-w)/2)
offset_x2 = random.uniform(-(612-w)/2, (612-w)/2)
offset_x3 = random.uniform(-(612-w)/2, (612-w)/2)
offset_x4 = random.uniform(-(612-w)/2, (612-w)/2)

offset_y1 = random.uniform(-(816-h)/2, (816-h)/2)
offset_y2 = random.uniform(-(816-h)/2, (816-h)/2)
offset_y3 = random.uniform(-(816-h)/2, (816-h)/2)
offset_y4 = random.uniform(-(816-h)/2, (816-h)/2)

jump_x = (612-w)/2
jump_y = (816-h)/2

pt1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
pt2 = np.float32([[0+jump_x+offset_x4, 0+jump_y+offset_y1], [jump_x+w+offset_x1, 0+offset_y2+jump_y], [jump_x+w+offset_x2, h+jump_y+offset_y3], [0+jump_x+offset_x3, h+jump_y+offset_y4]])
matrix = cv2.getPerspectiveTransform(pt1, pt2)
image = cv2.warpPerspective(img, matrix, (612, 816))

for p in pt2:
    cv2.circle(image,(int(p[0]),int(p[1])), 4, (0, 0, 255), -1)

cv2.imwrite('tmp.jpg', image)
cv2.imshow("Image", img)
cv2.imshow("Perspective transformation", image)
cv2.waitKey(0)