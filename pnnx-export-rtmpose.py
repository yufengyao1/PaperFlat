'''
Author: your name
Date: 2021-06-18 11:05:53
LastEditTime: 2024-02-02 14:01:19
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
import json
import copy
import pnnx
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.models.backbones.cspnext import CSPNeXt
model = CSPNeXt()
model.load_state_dict(torch.load('weights/rtmpose_14.pth', map_location='cpu'))
model.eval()
print(model)
x = torch.rand(1,3,224,224)
mod = torch.jit.trace(model, x)
mod.save("rtmpose.pt")
opt_model = pnnx.export(model, "rtmpose.pt", x)