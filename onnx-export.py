'''
Author: your name
Date: 2021-06-18 11:05:53
LastEditTime: 2023-11-06 14:34:12
LastEditors: yufengyao yufegnyao1@gmail.com
Description: In User Settings Edit
FilePath: \MultiClass\train.py
'''
import os
import cv2
import torch
import onnxruntime
import numpy as np
from torch.autograd import Variable
from PIL import Image
import os
import cv2
import torch
import numpy as np
from pfld import PFLDInference
import json
import copy
model = PFLDInference()
model.load_state_dict(torch.load('weights/pfld_1111-3.pth', map_location='cpu'))


x = torch.randn(1,3,224,224)
export_onnx_file = "paper1111.onnx"

torch.onnx.export(model,
                  x,
                  export_onnx_file,
                  opset_version=13,
                  export_params=True,
                  input_names=["input"],
                  output_names=["output"])

onnx_session = onnxruntime.InferenceSession(export_onnx_file,providers= ['CUDAExecutionProvider', 'CPUExecutionProvider'])



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


for file in val_files[0:50]:
    file = os.path.join('data/img', file)
    img = Image.open(file)
    w, h = img.size[0], img.size[1]
    # if h < w:
    #     continue


    frame = cv2.imread(file)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #转换成rgb
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB) #转换成rgb
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    # frame = frame[0:115, 0:155]
    frame = frame/255
    frame=(frame-0.5)/0.5 #转换到-1，1之间
    frame=frame.transpose(2,0,1)
    inputs = np.expand_dims(frame, axis=0)
    inputs=inputs.astype(np.float32)
    inputs = {onnx_session.get_inputs()[0].name: inputs}
    pred = onnx_session.run(None, inputs)
    pred=np.squeeze(pred)


    
    # img = transform(img)
    # inputs = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # inputs = inputs.to(device)
    # landmark = model(inputs)
    # landmark = landmark.tolist()[0]
    # print(landmark[-1])
    landmark_origin = copy.deepcopy(pred[0:8])

    scale = 1
    img_ori = Image.open(file)
    image = np.asarray(img_ori)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[0], image.shape[1]
    for i in range(8):
        if i % 2 == 0:
            pred[i] = pred[i]*w
        else:
            pred[i] = pred[i]*h
    
    points = []
    for i in range(4):
        points.append((int(pred[2*i]), int(pred[2*i+1])))
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
    points = (np.array(points)).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=(
        0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    scale = 610/frame_transformed.shape[0]
    frame_transformed = cv2.resize(
        frame_transformed, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    frame_transformed = frame_transformed[5:-5, 5:-5, :]

    cv2.imshow('paper1', image)
    cv2.imshow('paper2', frame_transformed)
    cv2.waitKey()


    
    # try:
    #     frame = cv2.imread(f)
    #     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #转换成rgb
    #     frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_LINEAR)
    #     frame = frame[0:115, 0:155]
    #     frame = frame/255
    #     frame=(frame-0.5)/0.5 #转换到-1，1之间
    #     frame=frame.transpose(2,0,1)
    #     inputs = np.expand_dims(frame, axis=0)
    #     inputs=inputs.astype(np.float32)
    #     inputs = {onnx_session.get_inputs()[0].name: inputs}
    #     pred = onnx_session.run(None, inputs)
    #     pred=np.squeeze(pred)
    #     pred=np.squeeze(pred)
    #     index_pred=np.argmax(pred)
    #     # pred=cal_softmax(pred)
    #     # print(pred)
        
    #     print(index_pred)
    # except Exception as ex:
    #     print(f)
    #     print(ex)
    #     continue