import os
import cv2
import copy
import json
import torch
import numpy as np
from PIL import Image
from pfld_448 import PFLDInference
from torchvision import transforms
from torch.autograd import Variable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = PFLDInference()
model.load_state_dict(torch.load('weights/pfld_448_0.pth', map_location='cpu'))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(
        (448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])



def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    rect = np.array(pts, dtype='float32')
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    if (heightA+heightB)>(widthA+widthB):
        maxHeight = int(maxWidth*4/3)
    else:
        maxHeight = int(maxWidth*3/4)
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), borderValue=(175, 175, 175))
    return warped


folder = "data/img_test"
files = os.listdir(folder)
files = [os.path.join(folder, f) for f in files]


for file in files[0:50]:
    image = cv2.imread(file)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    have_background=landmark[8]>landmark[9]
    landmark_origin = copy.deepcopy(landmark[0:8])
    
    exp_values=np.exp(np.array(landmark[8:10]))
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
