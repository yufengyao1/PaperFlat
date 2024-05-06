'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-13 11:25:41
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2024-01-13 13:27:59
FilePath: \PFLD-MINE\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import copy
import json
import torch
import numpy as np
from PIL import Image
from madepic import Madepic
from pfld_rle_448 import PFLDInference
from torchvision import transforms
from torch.autograd import Variable

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = PFLDInference().to(device)
model.load_state_dict(torch.load('weights/pfld_rle_448_3.pth', map_location='cpu'))

model = model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),  # ,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
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


val_files = []
json_folder = 'data/val'
json_files = os.listdir(json_folder)
for file in json_files:
    if file == '.DS_Store':
        continue
    with open(os.path.join(json_folder, file), 'r') as f:
        data = json.loads(f.read())
        val_files.append(data['imagePath'])


madepic = Madepic()
for i in range(50):
    img, _, _,_ = madepic.get_label()
    img_ori = img.copy()
    
    w, h = img.size[0], img.size[1]
    img_1 = transform(img)
    inputs = Variable(torch.unsqueeze(img_1, dim=0).float(), requires_grad=False)
    inputs = inputs.to(device)
    landmark = model(inputs)
    landmark = landmark.tolist()[0]
    
    out_direction=landmark[-4:]
    direction=np.argmax(np.array(out_direction))+1 #文字朝向
    # score = landmark[-1]
    # print(score)
    have_background=landmark[16]>landmark[17]
    landmark_origin = copy.deepcopy(landmark[0:16])
    landmark_origin=[landmark_origin[0],landmark_origin[1],landmark_origin[4],landmark_origin[5],landmark_origin[8],landmark_origin[9],landmark_origin[12],landmark_origin[13]]
    
    exp_values=np.exp(np.array(landmark[16:18]))
    sum_exp_values=np.sum(exp_values)
    softmax_values=exp_values/sum_exp_values
    softmax_values=softmax_values.tolist()
    
    print(softmax_values)
    print(direction)

    scale = 1
    image = np.asarray(img_ori)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[0], image.shape[1]
    landmark=copy.deepcopy(landmark_origin)
    for i in range(8):
        if i % 2 == 0:
            landmark[i] = landmark[i]*w
        else:
            landmark[i] = landmark[i]*h

    points = []
    for i in range(4):
        points.append((int(landmark[2*i]), int(landmark[2*i+1])))
    # for pt in points:
    #     cv2.circle(image, pt, 2, (0, 0, 255), cv2.FILLED,lineType=cv2.LINE_AA)
    frame_transformed = four_point_transform(image, points)

    scale = 400/image.shape[0]
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    h, w = image.shape[0], image.shape[1]
    for i in range(8):
        if i % 2 == 0:
            landmark_origin[i] = landmark_origin[i]*w
        else:
            landmark_origin[i] = landmark_origin[i]*h
    points = []
    for i in range(4):
        points.append((int(landmark_origin[2*i]), int(landmark_origin[2*i+1])))
    if have_background:
        cv2.circle(image, (int(points[0][0]), int(
            points[0][1])), 5, (255, 0, 0), cv2.FILLED, lineType=cv2.LINE_AA)
        points = (np.array(points)).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(
            0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        # zeros=np.zeros((image.shape),dtype=np.uint8)
        # mask=cv2.fillPoly(zeros,[points],(0,0,255))
        # image=cv2.addWeighted(image,0.9,mask,0.5,0)

    scale = 610/frame_transformed.shape[0]
    frame_transformed = cv2.resize(
        frame_transformed, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    frame_transformed = frame_transformed[5:-5, 5:-5, :]

    cv2.imshow('paper1', image)
    cv2.imshow('paper2', frame_transformed)
    print(file)
    k = cv2.waitKey()
    if k == 27:
        break
