'''
从人工标注得到前景图
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-13 11:25:41
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2023-11-02 16:59:13
FilePath: \PFLD-MINE\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import copy
import json
import torch
import uuid
import numpy as np
from PIL import Image
from pfld import PFLDInference
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=transforms.InterpolationMode.BICUBIC),#,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = pts
    rect = np.array(pts, dtype='float32')
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxHeight>maxWidth:
        maxHeight = int(maxWidth*4/3)
    else:
        maxHeight = int(maxWidth*3/4)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


train_files = []
json_folder = 'data/val'
json_files = os.listdir(json_folder)
for file in json_files:
    if file == '.DS_Store':
        continue
    with open(os.path.join(json_folder, file), 'r') as f:
        data = json.loads(f.read())

        # print(data["shapes"][0]["points"])
        try:
            train_files.append([data['imagePath'],data["shapes"][0]["points"]])
        except:
            continue
        

for file,points in train_files[0:]:
    file = os.path.join('data/img', file)
    image=cv2.imread(file)
  
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_ori=img.copy()
   

    frame_transformed = four_point_transform(image, points)
    # print(frame_transformed.shape)
    h,w,_=frame_transformed.shape
    frame_transformed=frame_transformed[10:h-10,10:w-10]
    filename=uuid.uuid4().hex
    cv2.imwrite('tmp/{0}.jpg'.format(filename),frame_transformed)

print("success")
    # cv2.imshow('paper1', frame_transformed)

    # cv2.waitKey()



