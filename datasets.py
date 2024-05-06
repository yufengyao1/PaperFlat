from random import random
from torch.utils.data import DataLoader
from torch.utils import data
import os
import random
import numpy as np
import cv2
import sys
import json
import uuid
import copy
from PIL import ImageDraw
from madepic import Madepic
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms.functional import crop
sys.path.append('..')


class WLFWDatasets(data.Dataset):
    def __init__(self, folder, transforms=None, mode='train',model_type="pfld"):
        self.model_type=model_type
        self.folder = folder
        self.data_list = []
        self.mode = mode
        # self.mode="val"
        self.transforms = transforms
        json_files = os.listdir(folder)
        self.current_index = 0
        self.made_pic = Madepic()
        for file in json_files:
            if file == '.DS_Store':
                continue

            with open(os.path.join(folder, file), 'r') as f:
                data = json.loads(f.read())
                if data['imageHeight'] < data['imageWidth']:
                    continue
                if len(data['shapes']) == 0:
                    continue
                if len(data['shapes'][0]['points']) != 4:
                    continue
                data.pop('imageData')
                # houzhui = data['imagePath'][-3:].upper()
                # if houzhui != "JPG":
                #     continue
                self.data_list.append(data)

    def __getitem__(self, index):
        if self.mode == "val":
            img = Image.open(os.path.join('data/img', self.data_list[index]['imagePath']))  # 使用真实标注数据
            points = copy.deepcopy(self.data_list[index]['shapes'][0]['points']) # 深拷贝，不然第二轮会使用上一轮偏移后的点
            w, h = img.size[0], img.size[1]
            landmark = []
            for i, p in enumerate(points):
                landmark.append(p[0]/w)
                landmark.append(p[1]/h)
            landmark.append(1)
            landmark = np.asarray(landmark, dtype=np.float32)
            img = self.transforms(img)
            return (img, landmark)

        img, points, have_foreground, direction = self.made_pic.get_label(self.model_type)  # 使用模拟图像

        # else: #使用真实图像
        #     real_index = np.random.randint(0, len(self.data_list)-1)
        # img = Image.open(os.path.join('data/img', self.data_list[real_index]['imagePath']))  # 使用真实标注数据
        # points = copy.deepcopy(self.data_list[real_index]['shapes'][0]['points'])  # 深拷贝，不然第二轮会使用上一轮偏移后的点

        # val = random.random()
        # back_h = img.height
        # back_w = img.width
        # if val < 0:  # 90度旋转
        #     # image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        #     img = img.transpose(Image.ROTATE_270)
        #     x1 = back_h-points[3][1]
        #     y1 = points[3][0]
        #     x2 = back_h-points[0][1]
        #     y2 = points[0][0]
        #     x3 = back_h-points[1][1]
        #     y3 = points[1][0]
        #     x4 = back_h-points[2][1]
        #     y4 = points[2][0]
        #     points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        #     # drawObject = ImageDraw.Draw(img)
        #     # drawObject.line([x1,y1,x2,y2],fill=10,width=5)
        #     # drawObject.line([x2,y2,x3,y3],fill=10,width=5)
        #     # drawObject.line([x3,y3,x4,y4],fill=10,width=5)
        #     # drawObject.line([x4,y4,x1,y1],fill=10,width=5)
        #     # # drawObject.point((x1,y1),(0,0,255))
        #     # drawObject.ellipse((x1,x2,x1+10,x2+10),fill="red")
        #     # img.save('tmp/{0}.jpg'.format(uuid.uuid1()))
        # elif val < 0:
        #     # image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     img = img.transpose(Image.ROTATE_90)
        #     x1 = points[1][1]
        #     y1 = back_w-points[1][0]
        #     x2 = points[2][1]
        #     y2 = back_w-points[2][0]
        #     x3 = points[3][1]
        #     y3 = back_w-points[3][0]
        #     x4 = points[0][1]
        #     y4 = back_w-points[0][0]
        #     points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        #     # drawObject = ImageDraw.Draw(img)

        #     # drawObject.line([x1,y1,x2,y2],fill=10,width=5)
        #     # drawObject.line([x2,y2,x3,y3],fill=10,width=5)
        #     # drawObject.line([x3,y3,x4,y4],fill=10,width=5)
        #     # drawObject.line([x4,y4,x1,y1],fill=10,width=5)
        #     # # drawObject.point((x1,y1),(0,0,255))
        #     # drawObject.ellipse((x1,x2,x1+20,x2+20),fill="red")

        #     # img.save('tmp/{0}.jpg'.format(uuid.uuid1()))
        # elif val < 0:
        #     img = img.transpose(Image.ROTATE_180)
        #     x1 = back_w-points[3][0]
        #     y1 = back_h-points[3][1]
        #     x2 = back_w-points[2][0]
        #     y2 = back_h-points[2][1]
        #     x3 = back_w-points[1][0]
        #     y3 = back_h-points[1][1]
        #     x4 = back_w-points[2][0]
        #     y4 = back_h-points[2][1]
        #     points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        #     # drawObject = ImageDraw.Draw(img)
        #     # drawObject.line([x1,y1,x2,y2],fill=10,width=5)
        #     # img.save('tmp/{0}.jpg'.format(uuid.uuid1()))

        w, h = img.size[0], img.size[1]

        # # output
        # drawObject = ImageDraw.Draw(img)
        # drawObject.line([points[0][0],points[0][1],points[1][0],points[1][1]],fill="red",width=5)
        # drawObject.line([points[1][0],points[1][1],points[2][0],points[2][1]],fill="yellow",width=5)
        # drawObject.line([points[2][0],points[2][1],points[3][0],points[3][1]],fill="blue",width=5)
        # drawObject.line([points[3][0],points[3][1],points[0][0],points[0][1]],fill="black",width=5)
        # img.save('tmp/{0}.jpg'.format(uuid.uuid1()))

        landmark = []
        if not have_foreground:  # 没有前景
            landmark = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        else:
            if self.model_type=="pfld":
                for p in points:
                    landmark.append(p[0]/img.size[0]) #pfld 0-1
                    landmark.append(p[1]/img.size[1]) #pfld 0-1
            elif self.model_type=="rtmpose":
                for p in points:
                    landmark.append(224*p[0]/img.size[0]) #rtmpose 0-224
                    landmark.append(224*p[1]/img.size[1]) #rtmpose 0-224
            landmark.extend([1, 0])

        if direction == 1:
            landmark.extend([1, 0, 0, 0])
        elif direction == 2:
            landmark.extend([0, 1, 0, 0])
        elif direction == 3:
            landmark.extend([0, 0, 1, 0])
        elif direction == 4:
            landmark.extend([0, 0, 0, 1])
        
        landmark = np.asarray(landmark, dtype=np.float32)
        img = self.transforms(img)
        return (img, landmark)

    def __len__(self):
        if self.mode == "train":
            return 5*len(self.data_list)
        else:
            return len(self.data_list)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    file_list = './data/test_data/list.txt'
    # wlfwdataset = WLFWDatasets(file_list)
    train_dataset = WLFWDatasets('data/val', transforms=transform)
    dataloader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    for img, landmark in dataloader:
        print(landmark.tolist())
