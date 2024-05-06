'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-02-27 13:17:53
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-03-02 14:42:45
FilePath: \图像平整\rotate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import random
from math import *
import numpy as np
 
# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    r=random.uniform(0,255)
    g=random.uniform(0,255)
    b=random.uniform(0,255)
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(r,g,b))

 
img = cv2.imread("C:\\Users\\yufen\\Desktop\\1.jpg")
# img = cv2.flip(img, 0)  # 垂直翻转
# img = cv2.flip(img, 1)  # 水平翻转
for i in range(1):

#直接进行高斯模糊操作
    img = cv2.GaussianBlur(img, (0,0), 2) #(0,0)高斯核：必须为正数和奇数，或者它们可以是零的。 最后一个参数，表示模糊程度，值越大越模糊

    # angle=random.uniform(-45,45)
    # imgRotation = rotate_bound_white_bg(img, angle)
    # imgRotation=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) #顺时针旋转
    # imgRotation=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE) #逆时针旋转
    # imgRotation=cv2.rotate(img,cv2.ROTATE_180) #逆时针旋转
    # imgRotation = cv2.flip(img, 0)  # 垂直翻转
    cv2.imshow("img",img)
    cv2.imshow("imgRotation",img)
    cv2.waitKey(0)