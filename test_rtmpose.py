import os
import cv2
import copy
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.models.backbones.cspnext import CSPNeXt
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = CSPNeXt()
model.load_state_dict(torch.load('weights/rtmpose_38.pth', map_location='cpu'))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def white_balance(frame):
    r, g, b = cv2.split(frame)
    # 计算图像的平均灰度值
    avg_gray = np.mean(frame)
    # 计算三个通道的灰度均值
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    # 计算三个通道的增益
    kr = avg_gray / avg_r
    kg = avg_gray / avg_g
    kb = avg_gray / avg_b
    # 调整图像的白平衡
    r = np.uint8(np.clip(r * kr, 0, 255))
    g = np.uint8(np.clip(g * kg, 0, 255))
    b = np.uint8(np.clip(b * kb, 0, 255))
    # 合并三个通道得到调整后的图像
    out_img = cv2.merge([r, g, b])
    return out_img

folder = "data/img_test"
files = os.listdir(folder)
files = [os.path.join(folder, f) for f in files]

simcc_label=SimCCLabel(input_size=(224, 224),sigma=(4.9, 5.66),simcc_split_ratio=2.0,normalize=False,use_dark=False)
for file in files[0:]:
    image = cv2.imread(file)
    # image=white_balance(image)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_ori = img.copy()
    w, h = img.size[0], img.size[1]
    img_1 = transform(img)
    inputs = Variable(torch.unsqueeze(img_1, dim=0).float(), requires_grad=False)
    inputs = inputs.to(device)
    t1=time.time()
    pred = model(inputs)
    t2=time.time()
    # print(t2-t1)
    landmark=simcc_label.decode(pred[0].detach().numpy(),pred[1].detach().numpy())
    scores=landmark[1].tolist()
    print(scores)
    
    landmark = (landmark[0].reshape(-1)/224).tolist()
    
    landmark_origin = copy.deepcopy(landmark[0:8])

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
    if True:
        cv2.circle(image, (int(points[0][0]), int(
            points[0][1])), 5, (255, 0, 0), cv2.FILLED, lineType=cv2.LINE_AA)
        points = (np.array(points)).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(
            0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

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
