import os
import cv2
import math
import copy
import uuid
import json
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms.functional import crop
from albumentations import Blur, Cutout, GridDropout, CoarseDropout


class Madepic:
    def __init__(self) -> None:
        self.folder = 'c:/data/paper'
        self.folder_back = 'c:/data/paper_back'
        self.files = os.listdir(self.folder)
        self.files_back = os.listdir(self.folder_back)
        self.base_mask = cv2.imread("data/img_black/black.jpg")
        self.base_mask = self.base_mask/255.0
        
        self.true_box_list_vertical = []  # 测试集中的真实框
        self.true_box_list_hor = []
        folder = 'data/train'
        json_files = os.listdir('data/train')
        for file in json_files:
            if file == '.DS_Store':
                continue
            with open(os.path.join(folder, file), 'r') as f:
                data = json.loads(f.read())
                if len(data['shapes']) == 0:
                    continue
                if len(data['shapes'][0]['points']) != 4:
                    continue
                points = data['shapes'][0]['points']
                img_height = data['imageHeight']
                img_width = data['imageWidth']
                x1 = points[0][0]
                y1 = points[0][1]
                x2 = points[1][0]
                y2 = points[1][1]
                x3 = points[2][0]
                y3 = points[2][1]
                x4 = points[3][0]
                y4 = points[3][1]
                if x1>x2 or x4>x3 or y1>y4 or y2>y3:
                    continue
                # if points[0][0]+points[0][1]>points[3][0]+points[3][1] or points[1][0]+points[1][1]>points[2][0]+points[2][1] or points[0][0]+points[0][1]>points[2][0]+points[2][1] :
                #     print(file)
                #     badfiles.append(file)
                for p in points:
                    p[0] = p[0]/img_width
                    p[1] = p[1]/img_height
                points.append(img_width)
                points.append(img_height)
                
                if img_height > img_width:
                    self.true_box_list_vertical.append(copy.deepcopy(points))
                else:
                    self.true_box_list_hor.append(copy.deepcopy(points))
        # for file in badfiles:
        #     os.remove(os.path.join(folder, file))
    
    def cal_angle(self, point_a, point_b, point_c):
        a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
        a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标
        if len(point_a) == len(point_b) == len(point_c) == 3:
            a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
        else:
            a_z, b_z, c_z = 0, 0, 0
        x1, y1, z1 = (a_x-b_x), (a_y-b_y), (a_z-b_z)
        x2, y2, z2 = (c_x-b_x), (c_y-b_y), (c_z-b_z)
        cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 +
                                                     z1**2) * (math.sqrt(x2**2 + y2**2 + z2**2)))  # 角点b的夹角余弦值
        B = math.degrees(math.acos(cos_b))  # 角点b的夹角值
        return B

    def rotate_random(self, image):
        try:
            val = random.random()
            if val > 0.95:  # 旋转背景
                angle = random.uniform(-45, 45)
                (fore_img_h, fore_img_w) = image.shape[:2]
                (cX, cY) = (fore_img_w // 2, fore_img_h // 2)
                # grab the rotation matrix (applying the negative of the
                # angle to rotate clockwise), then grab the sine and cosine
                # (i.e., the rotation components of the matrix)
                # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
                M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                # compute the new bounding dimensions of the image
                nW = int((fore_img_h * sin) + (fore_img_w * cos))
                nH = int((fore_img_h * cos) + (fore_img_w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY

                # perform the actual rotation and return the image
                # borderValue 缺失背景填充色彩，此处为白色，可自定义
                r = random.uniform(0, 255)
                g = random.uniform(0, 255)
                b = random.uniform(0, 255)
                return cv2.warpAffine(image, M, (nW, nH), borderValue=(r, g, b))
            else:
                return image
        except Exception as ex:
            print("error："+str(ex))
            return image

    def yy2img(self, img, base_mask):
        val = random.random()
        if val < 0.25:
            base_mask = cv2.rotate(base_mask, cv2.ROTATE_180)  # 180
        elif val < 0.5:
            base_mask = cv2.rotate(base_mask, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转
        elif val < 0.75:
            base_mask = cv2.rotate(base_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转
        height, width = img.shape[:2]
        base_mask = cv2.resize(base_mask, (width * 2, height*2), interpolation=cv2.INTER_CUBIC)
        H, W = base_mask.shape[:2]
        start_y = random.randint(0, H - height)
        end_y = start_y + height
        start_x = random.randint(0, W - width)
        end_x = start_x + width
        mask = base_mask[start_y:end_y, start_x:end_x]
        mask_img = img * mask
        return mask_img.astype(np.uint8)

    def random_move(self, x):
        offset = 0.05
        min_x = x-offset
        max_x = x+offset
        min_x = 0 if min_x < 0 else min_x
        max_x = 1 if max_x > 1 else max_x
        return random.uniform(min_x, max_x)
    
    def white_balance(self,frame):
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

    def get_label(self,model_type):
        direction = 1  # 旋转方向
        if model_type=="rtmpose": #必有前景
            have_foreground =True
        else: #pfld 
            have_foreground = random.random() > 0.1 # 是否有前景

        val = random.random()
        if val > 0.4:
            back_w = 1000
            back_h = 1333
        elif val > 0.3:
            back_w = 1000
            back_h = 1000
        else:
            back_w = 1000
            back_h = np.random.randint(1000, 1500)

        if have_foreground:  # 有前景
            index = random.randint(0, len(self.files)-1)
            img_1 = cv2.imread(os.path.join(self.folder, self.files[index]))

            # 随机裁剪前景图
            offset_h1 = random.randint(2, 10)
            offset_h2 = img_1.shape[0]-random.randint(2, 10)
            offset_w1 = random.randint(2, 10)
            offset_w2 = img_1.shape[1]-random.randint(2, 10)
            img_1 = img_1[offset_h1:offset_h2, offset_w1:offset_w2]

            img_1 = Blur(p=0.5)(image=img_1)['image']  # 高斯模糊前景图

            val = random.random()
            is_vertical = True
            if val < 0.25:
                img_1 = cv2.rotate(img_1, cv2.ROTATE_180)  # 180
                is_vertical = True
                direction = 3
            elif val < 0.5:
                img_1 = cv2.rotate(img_1, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转
                is_vertical = False
                direction = 2
            elif val < 0.75:
                img_1 = cv2.rotate(
                    img_1, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转
                is_vertical = False
                direction = 4
                
            fore_img_w, fore_img_h = img_1.shape[1], img_1.shape[0] #前景尺寸
            pt1 = np.float32([[0, 0], [fore_img_w, 0], [fore_img_w, fore_img_h], [0, fore_img_h]]) #pt1点集

            if random.random() > 0.6:
                while True:  # 计算前景投影点
                    try:
                        if is_vertical:  # 竖排
                            val = random.random()
                            if val >= 0.8:  # 贴边0-0.1
                                x1 = random.uniform(0, 0.1*back_w)
                                x4 = random.uniform(0, 0.1*back_w)
                                x2 = random.uniform(back_w*0.9, back_w)
                                x3 = random.uniform(back_w*0.9, back_w)

                                y1 = random.uniform(0, 0.1*back_h)
                                y2 = random.uniform(0, 0.1*back_h)
                                y3 = random.uniform(0.9*back_h, back_h)
                                y4 = random.uniform(0.9*back_h, back_h)
                            elif val >= 0.7:  # 贴边0.1-0.2
                                x1 = random.uniform(0.1*back_w, 0.2*back_w)
                                x4 = random.uniform(0.1*back_w, 0.2*back_w)
                                x2 = random.uniform(back_w*0.8, back_w*0.9)
                                x3 = random.uniform(back_w*0.8, back_w*0.9)

                                y1 = random.uniform(0.1*back_h, 0.2*back_h)
                                y2 = random.uniform(0.1*back_h, 0.2*back_h)
                                y3 = random.uniform(0.8*back_h, 0.9*back_h)
                                y4 = random.uniform(0.8*back_h, 0.9*back_h)
                            elif val >= 0.65:  # 贴边0.2-0.3
                                x1 = random.uniform(0.2*back_w, 0.3*back_w)
                                x4 = random.uniform(0.2*back_w, 0.3*back_w)
                                x2 = random.uniform(back_w*0.7, back_w*0.8)
                                x3 = random.uniform(back_w*0.7, back_w*0.8)

                                y1 = random.uniform(0.2*back_h, 0.3*back_h)
                                y2 = random.uniform(0.2*back_h, 0.3*back_h)
                                y3 = random.uniform(0.7*back_h, 0.8*back_h)
                                y4 = random.uniform(0.7*back_h, 0.8*back_h)
                            elif val >= 0.6:  # 贴边0-0.3
                                x1 = random.uniform(0, 0.2*back_w)
                                x4 = random.uniform(0, 0.2*back_w)
                                x2 = random.uniform(back_w*0.8, back_w)
                                x3 = random.uniform(back_w*0.8, back_w)

                                y1 = random.uniform(0, 0.25*back_h)
                                y2 = random.uniform(0, 0.25*back_h)
                                y3 = random.uniform(0.75*back_h, back_h)
                                y4 = random.uniform(0.75*back_h, back_h)
                            # elif val > 0.5:  # 上下
                            #     x1 = random.uniform(0, 0.2*back_w)
                            #     x4 = random.uniform(0, 0.2*back_w)
                            #     x2 = random.uniform(0.8*back_w, back_w)
                            #     x3 = random.uniform(0.8*back_w, back_w)

                            #     y1 = random.uniform(0, back_h-2*(x2-x1))
                            #     y2 = random.uniform(0, back_h-2*(x2-x1))
                            #     y3 = random.uniform(y2+x2-x1, y2+1.8*(x2-x1))
                            #     y4 = random.uniform(y1+x2-x1, y1+1.8*(x2-x1))
                            
                            elif val > 0.4:  # 竖拍梯形
                                x1 = random.uniform(2, 0.45*back_w)
                                x2 = random.uniform(0.55*back_w, back_w-2)
                                x3 = random.uniform(x2, back_w)
                                x4 = random.uniform(0, x1)

                                y1 = random.uniform(0, 0.3*back_h)
                                y2 = random.uniform(0, 0.3*back_h)
                                y3 = random.uniform(0.7*back_h, back_h)
                                y4 = random.uniform(0.7*back_h, back_h)
                            else:  # 随机
                                x1 = random.uniform(0, 0.5*back_w)
                                x2 = random.uniform(x1+0.45*back_w, back_w)
                                x4 = random.uniform(0, 0.5*back_w)
                                x3 = random.uniform(x4+0.45*back_w, back_w)

                                y1 = random.uniform(0, 0.5*back_h)
                                y4 = random.uniform(y1+0.45*back_h, back_h)
                                y2 = random.uniform(0, 0.5*back_h)
                                y3 = random.uniform(y2+0.45*back_h, back_h)
                        else:  # 横排
                            val = random.random()
                            if val >= 0.8:  # 梯形
                                x1 = random.uniform(0, 0.5*back_w)
                                x4 = random.uniform(0, 0.5*back_w)
                                x2 = random.uniform(x1+0.35*back_w, back_w)
                                x3 = random.uniform(x4+0.35*back_w, back_w)

                                y1 = random.uniform(0, 0.6*back_h)
                                y2 = random.uniform(0, 0.6*back_h)
                                y3 = random.uniform(y2+0.35*back_h, back_h)
                                y4 = random.uniform(y1+0.35*back_h, back_h)
                            elif val > 0.7:  # 横拍
                                x1 = random.uniform(0, 0.2*back_w)
                                x4 = random.uniform(0, 0.2*back_w)
                                x2 = random.uniform(0.8*back_w, back_w)
                                x3 = random.uniform(0.8*back_w, back_w)

                                y1 = random.uniform(0.2*back_h, 0.4*back_h)
                                tmp_offset = random.uniform(0, 0.2*back_h)
                                y2 = y1-tmp_offset
                                y4 = random.uniform(0.6*back_h, 0.8*back_h)
                                tmp_offset = random.uniform(0, 0.2*back_h)
                                y3 = y4+tmp_offset
                            elif val > 0.6:  # 横拍
                                x1 = random.uniform(0, 0.2*back_w)
                                x4 = random.uniform(0, 0.2*back_w)
                                x2 = random.uniform(0.8*back_w, back_w)
                                x3 = random.uniform(0.8*back_w, back_w)

                                y2 = random.uniform(0.2*back_h, 0.4*back_h)
                                tmp_offset = random.uniform(0, 0.2*back_h)
                                y1 = y2-tmp_offset+1
                                y3 = random.uniform(0.6*back_h, 0.8*back_h)
                                tmp_offset = random.uniform(0, 0.2*back_h)
                                y4 = y3+tmp_offset
                            elif val > 0.3:  # 横拍-框平移
                                x1 = random.uniform(0, 0.3*back_w)
                                x4 = random.uniform(0, 0.3*back_w)
                                x2 = random.uniform(0.7*back_w, back_w)
                                x3 = random.uniform(0.7*back_w, back_w)

                                rand_w = int(random.uniform(400, 800))
                                rand_y0 = int(random.uniform(150, back_h-rand_w-150))
                                y1 = random.uniform(-150, 150)+rand_y0
                                y2 = random.uniform(-150, 150)+rand_y0
                                y3 = rand_y0+random.uniform(-150, 150)+rand_w
                                y4 = rand_y0+random.uniform(-150, 150)+rand_w
                                
                            else:  # 随机
                                x1 = random.uniform(0, 0.3*back_w)
                                x2 = random.uniform(0.7*back_w, back_w)
                                x4 = random.uniform(0, 0.3*back_w)
                                x3 = random.uniform(0.7*back_w, back_w)

                                y1 = random.uniform(0, 0.6*back_h)
                                y4 = random.uniform(y1+0.35*back_h, back_h)
                                y2 = random.uniform(0, 0.6*back_h)
                                y3 = random.uniform(y2+0.35*back_h, back_h)
                        x1=0 if x1<0 else x1
                        x4=0 if x4<0 else x4
                        y1=0 if y1<0 else y1
                        y2=0 if y2<0 else y2
                        x2=back_w if x2>back_w else x2
                        x3=back_w if x3>back_w else x3
                        y3=back_h if y3>back_h else y3
                        y4=back_h if y4>back_h else y4
                        
                        if not (x1 < back_w/2 and x2 > back_w/2 and x3 > back_w/2 and x4 < back_w/2 and y1 < back_h/2 and y2 < back_h/2 and y3 > back_h/2 and y4 > back_h/2):  # 不在四个区分布
                            min_w = min(x2-x1, x3-x4)
                            min_h = min(y4-x1, y3-y2)

                            if direction == 1 or direction == 3:
                                if min_h/back_h < 0.25:  # 最小边小于整图边长的1/3
                                    continue
                            elif direction == 2 or direction == 4:
                                if min_w/back_w < 0.25:  # 最小边小于整图边长的1/3
                                    continue


                            k1 = (y1+y2-y3-y4)/(x1+x2-x3-x4)  # 纵轴倾斜角度不超过30度
                            if abs(k1) < 1.732:
                                continue
                            k2 = (y1+y4-y2-y3)/(x1+x4-x2-x3)  # 横轴倾斜角度不超过30度
                            if abs(k2) > 0.577:
                                continue

                        angle1 = self.cal_angle([x4, y4], [x1, y1], [x2, y2])
                        angle2 = self.cal_angle([x1, y1], [x2, y2], [x3, y3])
                        angle3 = self.cal_angle([x2, y2], [x3, y3], [x4, y4])
                        angle4 = self.cal_angle([x3, y3], [x4, y4], [x1, y1])

                        if angle1 >= 50 and angle1 < 140 and angle2 > 50 and angle2 < 140 and angle3 > 50 and angle3 < 140 and angle4 > 50 and angle4 < 140:
                            break
                    except Exception as ex:
                        print("e:"+str(ex))
                        continue
                pt2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]            
            else:  # 使用测试集中的真实点位
                while True:
                    if is_vertical:
                        box_index = random.randint(0, len(self.true_box_list_vertical)-1)
                        points = copy.deepcopy(self.true_box_list_vertical[box_index])
                    else:
                        box_index = random.randint(0, len(self.true_box_list_hor)-1)
                        points = copy.deepcopy(self.true_box_list_hor[box_index])
                    back_w = points[4]
                    back_h = points[5]
                    for p in points[0:4]:
                        p[0] = self.random_move(p[0])
                        p[1] = self.random_move(p[1])
                        p[0] = p[0]*back_w
                        p[1] = p[1]*back_h
                        
                    #随机offset
                    x_min=min(points[0][0],points[3][0])
                    x_max=max(points[1][0],points[2][0])
                    y_min=min(points[0][1],points[1][1])
                    y_max=max(points[2][1],points[3][1])
                    offsetx=random.randint(-1*int(x_min), int(back_w-x_max-1)) if int(back_w-x_max-1)>0 else 0
                    offsety=random.randint(-1*int(y_min), int(back_h-y_max-1)) if int(back_h-y_max-1)>0 else 0
                    for p in points[0:4]:
                        p[0]+=offsetx
                        p[1]+=offsety
                        
                    x1 = points[0][0]
                    y1 = points[0][1]
                    x2 = points[1][0]
                    y2 = points[1][1]
                    x3 = points[2][0]
                    y3 = points[2][1]
                    x4 = points[3][0]
                    y4 = points[3][1]
                    if x1>x2 or x4>x3 or y1>y4 or y2>y3:
                        continue
                    points = points[:4] #去掉多余的两个w,h值
                    pt2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]) #投影后的点
                    break

            
            many_paper = random.random() > 0.8  # 多张纸叠加
            if many_paper:
                if random.random() > 0.5:
                    index2 = random.randint(0, len(self.files)-1)
                    img_2 = cv2.imread(os.path.join(
                        self.folder, self.files[index2]))
                else:
                    img_2 = img_1.copy()

                val = random.random()
                if val > 0.6:
                    paper_offset = 200
                    x1_back = x1+np.random.randint(-paper_offset, paper_offset)
                    x2_back = x2+np.random.randint(-paper_offset, paper_offset)
                    x3_back = x3+np.random.randint(-paper_offset, paper_offset)
                    x4_back = x4+np.random.randint(-paper_offset, paper_offset)
                    y1_back = y1+np.random.randint(-paper_offset, paper_offset)
                    y2_back = y2+np.random.randint(-paper_offset, paper_offset)
                    y3_back = y3+np.random.randint(-paper_offset, paper_offset)
                    y4_back = y4+np.random.randint(-paper_offset, paper_offset)
                    pt2_back = np.float32([[x1_back, y1_back], [x2_back, y2_back], [
                                          x3_back, y3_back], [x4_back, y4_back]])
                    pt1_back = np.float32([[0, 0], [img_2.shape[1], 0], [
                                          img_2.shape[1], img_2.shape[0]], [0, img_2.shape[0]]])
                    matrix_back = cv2.getPerspectiveTransform(
                        pt1_back, pt2_back)  # 被遮图变换矩阵
                elif val > 0.4:
                    paper_offset_x = np.random.randint(-1*fore_img_w//2, fore_img_w//2)
                    paper_offset_y = np.random.randint(-1*fore_img_h//2, fore_img_h//2)
                    x1_back = x1+paper_offset_x
                    x2_back = x2+paper_offset_x
                    x3_back = x3+paper_offset_x
                    x4_back = x4+paper_offset_x
                    y1_back = y1+paper_offset_y
                    y2_back = y2+paper_offset_y
                    y3_back = y3+paper_offset_y
                    y4_back = y4+paper_offset_y
                    pt2_back = np.float32([[x1_back, y1_back], [x2_back, y2_back], [
                                          x3_back, y3_back], [x4_back, y4_back]])
                    pt1_back = np.float32([[0, 0], [img_2.shape[1], 0], [
                                          img_2.shape[1], img_2.shape[0]], [0, img_2.shape[0]]])
                    matrix_back = cv2.getPerspectiveTransform(
                        pt1_back, pt2_back)  # 被遮图变换矩阵
                else:
                    center_x = (x1+x2+x3+x4)//4
                    center_y = (y1+y2+y3+y4)//4

                    angle = np.random.randint(-90, 90)
                    x1_back, y1_back = self.rotate_points(
                        x1, y1, center_x, center_y, fore_img_w, fore_img_h, angle)
                    x2_back, y2_back = self.rotate_points(
                        x2, y2, center_x, center_y, fore_img_w, fore_img_h, angle)
                    x3_back, y3_back = self.rotate_points(
                        x3, y3, center_x, center_y, fore_img_w, fore_img_h, angle)
                    x4_back, y4_back = self.rotate_points(
                        x4, y4, center_x, center_y, fore_img_w, fore_img_h, angle)
                    pt2_back = np.float32([[x1_back, y1_back], [x2_back, y2_back], [
                                          x3_back, y3_back], [x4_back, y4_back]])
                    pt1_back = np.float32([[0, 0], [img_2.shape[1], 0], [
                                          img_2.shape[1], img_2.shape[0]], [0, img_2.shape[0]]])
                    matrix_back = cv2.getPerspectiveTransform(
                        pt1_back, pt2_back)  # 被遮图变换矩阵

                border_value = (56, 154, 10)
                image_2 = cv2.warpPerspective(
                    img_2, matrix_back, (back_w, back_h),  borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
                mask_2 = (image_2[:, :, 0] == border_value[0]) & (
                    image_2[:, :, 1] == border_value[1]) & (image_2[:, :, 2] == border_value[2])

            border_value = (56, 154, 10)
            matrix = cv2.getPerspectiveTransform(pt1, pt2)  # 前景变换矩阵
            image = cv2.warpPerspective(img_1, matrix, (back_w, back_h),  borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)  # 前景变换
            
            #随机裁剪边角
            angle_cut=False
            if random.random() > 0.8:
                angle_cut=True
                percent_1,percent_2=random.uniform(0.03, 0.25),random.uniform(0.03, 0.25)
                cross_x1=(x2-x1)*percent_1+x1
                cross_y1=(y2-y1)*percent_1+y1
                cross_x2=(x4-x1)*percent_2+x1
                cross_y2=(y4-y1)*percent_2+y1
                points_polygon=np.array([[[int(cross_x1),int(cross_y1)],[int(cross_x2),int(cross_y2)],[int(x1),int(y1)]]])
                image=cv2.fillPoly(image,points_polygon,border_value)
            mask = (image[:, :, 0] == border_value[0]) & (image[:, :, 1] == border_value[1]) & (image[:, :, 2] == border_value[2]) #背景mask
            
            if random.random() > 0.95:  # 随机纯色背景
                val = random.random()
                if val > 0.8:
                    b, g, r = random.randint(0, 30), random.randint(
                        0, 30), random.randint(0, 30)
                elif val > 0.5:
                    b, g, r = random.randint(180, 255), random.randint(
                        180, 255), random.randint(180, 255)
                else:
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                image_back = np.zeros((back_h, back_w, 3), dtype=np.uint8)
                image_back[:] = [b, g, r]
                if many_paper:
                    image_2[mask_2] = image_back[mask_2]
                    image[mask] = image_2[mask]
                else:
                    image[mask] = image_back[mask]
                # image = cv2.warpPerspective(img_1, matrix, (back_w, back_h),  borderMode=cv2.BORDER_CONSTANT, borderValue=(b, g, r))
            else:  # 图片背景
                index = random.randint(0, len(self.files_back)-1)
                image_back = cv2.imread(os.path.join(
                    self.folder_back, self.files_back[index]))

                # 随机裁剪背景图
                offset_h1 = random.randint(0, int(image_back.shape[0]/3))
                offset_h2 = image_back.shape[0] - \
                    random.randint(0, int(image_back.shape[0]/3))
                offset_w1 = random.randint(0, int(image_back.shape[1]/3))
                offset_w2 = image_back.shape[1] - \
                    random.randint(0, int(image_back.shape[1]/3))
                image_back = image_back[offset_h1:offset_h2,
                                        offset_w1:offset_w2]

                # 翻转背景图
                val = random.random()
                if val < 0.25:
                    image_back = cv2.rotate(image_back, cv2.ROTATE_180)  # 180
                elif val < 0.5:
                    image_back = cv2.rotate(
                        image_back, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转
                elif val < 0.75:
                    image_back = cv2.rotate(
                        image_back, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转

                image_back = self.rotate_random(image_back)

                # 随机裁剪背景图到指定比例4:3
                b_h, b_w = image_back.shape[0], image_back.shape[1]
                if b_h != back_h and b_w != back_w:
                    if b_h/b_w > back_h/back_w:
                        need_h = int(b_w*back_h/back_w)
                        offset_h = b_h-need_h
                        if offset_h > 1:
                            rand_h = random.randint(0, offset_h)
                            image_back = image_back[rand_h:rand_h+need_h, :, :]
                    else:
                        need_w = int(b_h*back_w/back_h)
                        offset_w = b_w-need_w
                        if offset_w > 1:
                            rand_w = random.randint(0, offset_w)
                            image_back = image_back[:, rand_w:rand_w+need_w, :]

                image_back = cv2.resize(
                    image_back, (image.shape[1], image.shape[0]))
                image_back = Blur(p=0.5)(image=image_back)['image']  # 高斯模糊背景图

                if many_paper:
                    image_2[mask_2] = image_back[mask_2]
                    image[mask] = image_2[mask]
                else:
                    image[mask] = image_back[mask]

            #左上角处理
            val=random.random()
            if val>0.7 and not angle_cut: #左上角翻页
                percent_1,percent_2=random.uniform(0.05, 0.25),random.uniform(0.05, 0.25)
                cross_x1=(x2-x1)*percent_1+x1
                cross_y1=(y2-y1)*percent_1+y1
                cross_x2=(x4-x1)*percent_2+x1
                cross_y2=(y4-y1)*percent_2+y1
                x2_new,y2_new=self.cal_fanye_point(cross_x1,cross_y1,cross_x2,cross_y2,x2,y2)
                x3_new,y3_new=self.cal_fanye_point(cross_x1,cross_y1,cross_x2,cross_y2,x3,y3)
                x4_new,y4_new=self.cal_fanye_point(cross_x1,cross_y1,cross_x2,cross_y2,x4,y4)
                offset=random.randint(-10,10) #随机调整折线点
                cross_x1+=offset
                offset=random.randint(-10,10)
                cross_y1+=offset
                offset=random.randint(-10,10)
                cross_x2+=offset
                offset=random.randint(-10,10)
                cross_y2+=offset
                points_polygon=np.array([[[int(cross_x1),int(cross_y1)],[int(cross_x2),int(cross_y2)],[int(x4_new),int(y4_new)],[int(x3_new),int(y3_new)],[int(x2_new),int(y2_new)]]])
                r=random.randint(200,255)
                g=random.randint(200,255)
                b=random.randint(200,255)
                image=cv2.fillPoly(image,points_polygon,(r,g,b))
        else:  # 没有前景
            if random.random() > 0.8:  # 随机纯色背景
                val = random.random()
                if val > 0.8:
                    b, g, r = random.randint(0, 30), random.randint(
                        0, 30), random.randint(0, 30)
                elif val > 0.5:
                    b, g, r = random.randint(180, 255), random.randint(
                        180, 255), random.randint(180, 255)
                else:
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                image = np.zeros((back_h, back_w, 3), dtype=np.uint8)
                image[:] = [b, g, r]
                fore_img_h, fore_img_w = back_h, back_w
                points = [[0, 0], [fore_img_w, 0], [fore_img_w, fore_img_h], [0, fore_img_h]]
            else:  # 图片背景
                index = random.randint(0, len(self.files_back)-1)
                image_back = cv2.imread(os.path.join(
                    self.folder_back, self.files_back[index]))

                # 随机裁剪背景图
                offset_h1 = random.randint(0, int(image_back.shape[0]/3))
                offset_h2 = image_back.shape[0] - \
                    random.randint(0, int(image_back.shape[0]/3))
                offset_w1 = random.randint(0, int(image_back.shape[1]/3))
                offset_w2 = image_back.shape[1] - \
                    random.randint(0, int(image_back.shape[1]/3))
                image_back = image_back[offset_h1:offset_h2,
                                        offset_w1:offset_w2]

                # 翻转背景图
                val = random.random()
                if val < 0.25:
                    image_back = cv2.rotate(image_back, cv2.ROTATE_180)  # 180
                elif val < 0.5:
                    image_back = cv2.rotate(
                        image_back, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转
                elif val < 0.75:
                    image_back = cv2.rotate(
                        image_back, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转

                image_back = self.rotate_random(image_back)

                # 随机裁剪背景图到指定比例4:3
                b_h, b_w = image_back.shape[0], image_back.shape[1]
                if b_h != back_h and b_w != back_w:
                    if b_h/b_w > back_h/back_w:
                        need_h = int(b_w*back_h/back_w)
                        offset_h = b_h-need_h
                        if offset_h > 1:
                            rand_h = random.randint(0, offset_h)
                            image_back = image_back[rand_h:rand_h+need_h, :, :]
                    else:
                        need_w = int(b_h*back_w/back_h)
                        offset_w = b_w-need_w
                        if offset_w > 1:
                            rand_w = random.randint(0, offset_w)
                            image_back = image_back[:, rand_w:rand_w+need_w, :]

                image_back = cv2.resize(image_back, (back_w, back_h))
                image_back = Blur(p=0.5)(image=image_back)['image']  # 高斯模糊背景图

                image = image_back
                fore_img_h, fore_img_w = back_h, back_w
                points = [[0, 0], [fore_img_w, 0], [fore_img_w, fore_img_h], [0, fore_img_h]]

        if have_foreground:  # 有前景才旋转，不然无前景的1变成-4
            val = random.random()  # 旋转整张图，不然全是高大于宽
            if val < 0.25:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                x1 = back_h-points[3][1]
                y1 = points[3][0]
                x2 = back_h-points[0][1]
                y2 = points[0][0]
                x3 = back_h-points[1][1]
                y3 = points[1][0]
                x4 = back_h-points[2][1]
                y4 = points[2][0]
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                direction += 1
                direction = 1 if direction == 5 else direction
            elif val < 0.5:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                x1 = points[1][1]
                y1 = back_w-points[1][0]
                x2 = points[2][1]
                y2 = back_w-points[2][0]
                x3 = points[3][1]
                y3 = back_w-points[3][0]
                x4 = points[0][1]
                y4 = back_w-points[0][0]
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

                direction -= 1
                direction = 4 if direction == 0 else direction
        
        if random.random() >0.5: #添加阴影-不可少
            image=self.yy2img(image,self.base_mask.copy()) 
        
        val = random.random()  # 添加椒盐噪声
        if val > 0.95:
            # 设置添加椒盐噪声的数目比例
            s_vs_p = 0.5
            # 设置添加噪声图像像素的数目
            amount = random.uniform(0.001, 0.015)
            noisy_img = np.copy(image)
            # 添加salt噪声
            num_salt = np.ceil(amount * image.size * s_vs_p)
            # 设置添加噪声的坐标位置
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            noisy_img[coords[0], coords[1], :] = [255, 255, 255]
            # 添加pepper噪声
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            # 设置添加噪声的坐标位置
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            noisy_img[coords[0], coords[1], :] = [0, 0, 0]
            image = noisy_img

        fill_value = np.random.randint(0, 255)
        image = CoarseDropout(max_height=100, max_width=100, p=0.1, fill_value=fill_value)(image=image)['image']  # 8个黑块

        image=GridDropout(fill_value=fill_value,p=0.05)(image=image)['image'] #黑白相间
        
        image=Blur(p=0.2)(image=image)['image'] #高斯模糊整张图
        
        # image=self.white_balance(image)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if random.random() > 0.8 and have_foreground and model_type=="pfld": #rtmpose 不裁剪
            image, points = self.cut_img(points, image, direction)

        # # output
        # if have_foreground:# and not use_true_box:
        # image_tmp=np.array(image.copy())
        # cv2.circle(image_tmp, (int(points[0][0]),int(points[0][1])), 5, (255, 0, 0), cv2.FILLED,lineType=cv2.LINE_AA)
        # cv2.putText(image_tmp, str(direction), (150, 150), cv2.FONT_HERSHEY_PLAIN,10, (0, 0, 255), 2)
        # pts = (np.array(points,dtype=int)).reshape((-1, 1, 2))
        # cv2.polylines(image_tmp, [pts], isClosed=True,color=(0, 0, 255), thickness=2,lineType=cv2.LINE_AA)
        # cv2.imwrite('tmp/{0}.jpg'.format(uuid.uuid1()),image_tmp)

        return image, points, have_foreground, direction

    def cal_fanye_point(self,x1,y1,x2,y2,x3,y3): #点关于任意对称轴的镜像计算
        y1=-y1
        y2=-y2
        y3=-y3
        theta=math.atan2(y2-y1,x2-x1)-math.atan2(y3-y1,x3-x1)
        theta2=theta*2
        x=x1+(x3-x1)*math.cos(theta2)-(y3-y1)*math.sin(theta2)
        y=y1+(x3-x1)*math.sin(theta2)+(y3-y1)*math.cos(theta2)
        return x,-y
    
    def cut_img(self, points, img, direction):
        back_img_w, back_img_h = img.size[0], img.size[1]
        while True:
            offset_x1, offset_y1, offset_x2, offset_y2 = 0, 0, 0, 0
            cut_left_right=0.3
            cut_top_bottom=0.1
            if direction == 1:
                crop_percent_top = cut_top_bottom
                crop_percent_bottom = cut_top_bottom
                crop_percent_left = cut_left_right
                crop_percent_right = cut_left_right
            elif direction == 3:
                crop_percent_top = cut_top_bottom
                crop_percent_bottom = cut_top_bottom
                crop_percent_left = cut_left_right
                crop_percent_right = cut_left_right
            elif direction == 2:
                crop_percent_top = cut_left_right
                crop_percent_bottom = cut_left_right
                crop_percent_left = cut_top_bottom
                crop_percent_right = cut_top_bottom
            elif direction == 4:
                crop_percent_top = cut_left_right
                crop_percent_bottom = cut_left_right
                crop_percent_left = cut_top_bottom
                crop_percent_right = cut_top_bottom

            if random.random() < 0.5:
                left1 = min(int(back_img_w*crop_percent_left), int(0.5 *
                            points[1][0]+0.5*points[0][0])-0.25*(points[1][0]-points[0][0]))
                left2 = min(int(back_img_w*crop_percent_left), int(0.5 *
                            points[2][0]+0.5*points[3][0])-0.25*(points[2][0]-points[3][0]))
                left = min(left1, left2)
                left = 0 if left < 2 else left
                offset_x1 = 0 if left < 2 else np.random.randint(0, left)

            if random.random() < 0.5:
                right1 = min(int(back_img_w*crop_percent_right), int(back_img_w-(0.5 *
                             points[1][0]+0.5*points[0][0])+0.25*(points[1][0]-points[0][0])))
                right2 = min(int(back_img_w*crop_percent_right), int(back_img_w-(0.5 *
                             points[2][0]+0.5*points[3][0])+0.25*(points[2][0]-points[3][0])))
                right = min(right1, right2)
                right = 0 if right <= 0 else right
                offset_x2 = 0 if right < 2 else np.random.randint(0, right)

            if random.random() < 0.5:
                top1 = min(int(back_img_h*crop_percent_top), int(0.5 *
                           points[0][1]+0.5*points[3][1])-0.25*(points[3][1]-points[0][1]))
                top2 = min(int(back_img_h*crop_percent_top), int(0.5 *
                           points[1][1]+0.5*points[2][1])-0.25*(points[2][1]-points[1][1]))
                top = min(top1, top2)
                top = 0 if top <= 0 else top
                offset_y1 = 0 if top < 2 else np.random.randint(0, top)

            if random.random() < 0.5:
                bottom1 = min(int(
                    back_img_h*crop_percent_bottom), int(back_img_h-(0.5*points[0][1]+0.5*points[3][1])+0.25*(points[3][1]-points[0][1])))
                bottom2 = min(int(
                    back_img_h*crop_percent_bottom), int(back_img_h-(0.5*points[1][1]+0.5*points[2][1])+0.25*(points[2][1]-points[1][1])))
                bottom = min(bottom1, bottom2)
                bottom = 0 if bottom <= 0 else bottom
                offset_y2 = 0 if bottom < 2 else np.random.randint(0, bottom)

            x1 = points[0][0]-offset_x1
            y1 = points[0][1]-offset_y1
            x2 = points[1][0]-offset_x1
            y2 = points[1][1]-offset_y1
            x3 = points[2][0]-offset_x1
            y3 = points[2][1]-offset_y1
            x4 = points[3][0]-offset_x1
            y4 = points[3][1]-offset_y1

            new_w = back_img_w-offset_x1-offset_x2 #裁剪后尺寸
            new_h = back_img_h-offset_y1 - offset_y2 #裁剪后尺寸
            
            if new_h/new_w > 2.5 or new_h/new_w < 0.4:  # 裁剪以后比例失调
                continue
            
            cut_in = 2
            if cut_in == 1:  # 向内延展
                if x1 < 0 and x4 < 0:  # left out
                    rate1 = -1*x1/x2
                    rate4 = -1*x4/x3
                    if rate1 < rate4:  # point1 move
                        # if abs(y3-y4)>2:
                        k4 = (y3-y4)/(x3-x4)
                        b4 = y4-k4*x4
                        y4 = b4
                        x4 = 0

                        lambdax = rate4
                        x1 = int((x1+lambdax*x2)/(1+lambdax))
                        y1 = int((y1+lambdax*y2)/(1+lambdax))
                    else:  # point4 move
                        # if abs(y1-y2)>2:
                        k1 = (y2-y1)/(x2-x1)
                        b1 = y1-k1*x1
                        y1 = b1
                        x1 = 0

                        lambdax = rate1
                        x4 = int((x4+lambdax*x3)/(1+lambdax))
                        y4 = int((y4+lambdax*y3)/(1+lambdax))
                if y4 > new_h and y3 > new_h:  # bottom out
                    rate4 = (y4-new_h)/(new_h-y1)
                    rate3 = (y3-new_h)/(new_h-y2)
                    if rate4 < rate3:  # point4 move
                        # y3=new_h
                        if abs(x3-x2) > 2:
                            k3 = (y3-y2)/(x3-x2)
                            b3 = y3-k3*x3
                            x3 = (new_h-b3)/k3
                        y3 = new_h

                        lambdax = rate3
                        x4 = int((x4+lambdax*x1)/(1+lambdax))
                        y4 = int((y4+lambdax*y1)/(1+lambdax))
                    else:  # point3 move
                        # y4=new_h
                        if abs(x4-x1) > 2:
                            k4 = (y4-y1)/(x4-x1)
                            b4 = y4-k4*x4
                            x4 = (new_h-b4)/k4
                        y4 = new_h

                        lambdax = rate4
                        x3 = int((x3+lambdax*x2)/(1+lambdax))
                        y3 = int((y3+lambdax*y2)/(1+lambdax))
                if x2 > new_w and x3 > new_w:  # right out
                    rate2 = (x2-new_w)/(new_w-x1)
                    rate3 = (x3-new_w)/(new_w-x4)
                    if rate2 < rate3:  # point2 move
                        if abs(y3-y4) > 2:
                            k3 = (y3-y4)/(x3-x4)
                            b3 = y3-k3*x3
                            x3 = new_w
                            y3 = k3*x3+b3
                        x3 = new_w

                        lambdax = rate3
                        x2 = int((x2+lambdax*x1)/(1+lambdax))
                        y2 = int((y2+lambdax*y1)/(1+lambdax))
                    else:  # point3 move
                        if abs(y2-y1) > 2:
                            k2 = (y2-y1)/(x2-x1)
                            b2 = y2-k2*x2
                            x2 = new_w
                            y2 = k2*x2+b2
                        x2 = new_w

                        lambdax = rate2
                        x3 = int((x3+lambdax*x4)/(1+lambdax))
                        y3 = int((y3+lambdax*y4)/(1+lambdax))
                if y1 < 0 and y2 < 0:  # top out
                    rate1 = -1*y1/y4
                    rate2 = -1*y2/y3
                    if rate1 < rate2:  # point1 move
                        # y2=0
                        if abs(x3-x2) > 2:
                            k2 = (y3-y2)/(x3-x2)
                            b2 = y2-k2*x2
                            x2 = (0-b2)/k2
                        y2 = 0

                        lambdax = rate2
                        x1 = (x1+lambdax*x4)/(1+lambdax)
                        y1 = (y1+lambdax*y4)/(1+lambdax)
                    else:  # point2 move
                        if abs(x4-x1) > 2:
                            k1 = (y4-y1)/(x4-x1)
                            b1 = y1-k1*x1
                            x1 = (0-b1)/k1
                        y1 = 0

                        lambdax = rate1
                        x2 = int((x2+lambdax*x3)/(1+lambdax))
                        y2 = int((y2+lambdax*y3)/(1+lambdax))
            elif cut_in == 2:  # 向外延展
                if x1 < 0 and x4 < 0:  # left out
                    rate1 = -1*x1/x2
                    rate4 = -1*x4/x3
                    if rate1 > rate4:  # point1 move
                        k4 = (y3-y4)/(x3-x4)
                        b4 = y4-k4*x4
                        y4 = b4
                        x4 = 0

                        lambdax = rate4
                        x1 = int((x1+lambdax*x2)/(1+lambdax))
                        y1 = int((y1+lambdax*y2)/(1+lambdax))
                    else:  # point4 move
                        # if abs(y1-y2)>2:
                        k1 = (y2-y1)/(x2-x1)
                        b1 = y1-k1*x1
                        y1 = b1
                        x1 = 0

                        lambdax = rate1
                        x4 = int((x4+lambdax*x3)/(1+lambdax))
                        y4 = int((y4+lambdax*y3)/(1+lambdax))
                if y4 > new_h and y3 > new_h:  # bottom out
                    rate4 = (y4-new_h)/(new_h-y1)
                    rate3 = (y3-new_h)/(new_h-y2)
                    if rate4 > rate3:  # point4 move
                        if abs(x3-x2) > 2:
                            k3 = (y3-y2)/(x3-x2)
                            b3 = y3-k3*x3
                            x3 = (new_h-b3)/k3
                        y3 = new_h

                        lambdax = rate3
                        x4 = int((x4+lambdax*x1)/(1+lambdax))
                        y4 = int((y4+lambdax*y1)/(1+lambdax))
                    else:  # point3 move
                        # y4=new_h
                        if abs(x4-x1) > 2:
                            k4 = (y4-y1)/(x4-x1)
                            b4 = y4-k4*x4
                            x4 = (new_h-b4)/k4
                        y4 = new_h

                        lambdax = rate4
                        x3 = int((x3+lambdax*x2)/(1+lambdax))
                        y3 = int((y3+lambdax*y2)/(1+lambdax))
                if x2 > new_w and x3 > new_w:  # right out
                    rate2 = (x2-new_w)/(new_w-x1)
                    rate3 = (x3-new_w)/(new_w-x4)
                    if rate2 > rate3:  # point2 move
                        if abs(y3-y4) > 2:
                            k3 = (y3-y4)/(x3-x4)
                            b3 = y3-k3*x3
                            x3 = new_w
                            y3 = k3*x3+b3
                        x3 = new_w

                        lambdax = rate3
                        x2 = int((x2+lambdax*x1)/(1+lambdax))
                        y2 = int((y2+lambdax*y1)/(1+lambdax))
                    else:  # point3 move
                        if abs(y2-y1) > 2:
                            k2 = (y2-y1)/(x2-x1)
                            b2 = y2-k2*x2
                            x2 = new_w
                            y2 = k2*x2+b2
                        x2 = new_w

                        lambdax = rate2
                        x3 = int((x3+lambdax*x4)/(1+lambdax))
                        y3 = int((y3+lambdax*y4)/(1+lambdax))
                if y1 < 0 and y2 < 0:  # top out
                    rate1 = -1*y1/y4
                    rate2 = -1*y2/y3
                    if rate1 > rate2:  # point1 move
                        if abs(x3-x2) > 2:
                            k2 = (y3-y2)/(x3-x2)
                            b2 = y2-k2*x2
                            x2 = (0-b2)/k2
                        y2 = 0

                        lambdax = rate2
                        x1 = (x1+lambdax*x4)/(1+lambdax)
                        y1 = (y1+lambdax*y4)/(1+lambdax)
                    else:  # point2 move
                        if abs(x4-x1) > 2:
                            k1 = (y4-y1)/(x4-x1)
                            b1 = y1-k1*x1
                            x1 = (0-b1)/k1
                        y1 = 0

                        lambdax = rate1
                        x2 = int((x2+lambdax*x3)/(1+lambdax))
                        y2 = int((y2+lambdax*y3)/(1+lambdax))
            else:  # 向内截断
                if x1 < 0 and x4 < 0:  # left out
                    k4 = (y3-y4)/(x3-x4)
                    b4 = y4-k4*x4
                    y4 = b4
                    x4 = 0

                    k1 = (y2-y1)/(x2-x1)
                    b1 = y1-k1*x1
                    y1 = b1
                    x1 = 0

                if y4 > new_h and y3 > new_h:  # bottom out
                    if abs(x3-x2) > 2:
                        k3 = (y3-y2)/(x3-x2)
                        b3 = y3-k3*x3
                        x3 = (new_h-b3)/k3
                    y3 = new_h

                    if abs(x4-x1) > 2:
                        k4 = (y4-y1)/(x4-x1)
                        b4 = y4-k4*x4
                        x4 = (new_h-b4)/k4
                    y4 = new_h
                if x2 > new_w and x3 > new_w:  # right out
                    if abs(y3-y4) > 2:
                        k3 = (y3-y4)/(x3-x4)
                        b3 = y3-k3*x3
                        x3 = new_w
                        y3 = k3*x3+b3
                    x3 = new_w

                    if abs(y2-y1) > 2:
                        k2 = (y2-y1)/(x2-x1)
                        b2 = y2-k2*x2
                        x2 = new_w
                        y2 = k2*x2+b2
                    x2 = new_w

                if y1 < 0 and y2 < 0:  # top out
                    if abs(x3-x2) > 2:
                        k2 = (y3-y2)/(x3-x2)
                        b2 = y2-k2*x2
                        x2 = (0-b2)/k2
                    y2 = 0

                    if abs(x4-x1) > 2:
                        k1 = (y4-y1)/(x4-x1)
                        b1 = y1-k1*x1
                        x1 = (0-b1)/k1
                    y1 = 0

            max_w = max(x2-x1, x3-x4)
            max_h = max(y4-x1, y3-y2)

            
            if max_w/new_w < 0.25 or max_h/new_h < 0.25:
                continue
            
            points[0][0] = x1
            points[0][1] = y1
            points[1][0] = x2
            points[1][1] = y2
            points[2][0] = x3
            points[2][1] = y3
            points[3][0] = x4
            points[3][1] = y4

            img = crop(img, offset_y1, offset_x1, new_h, new_w)
            break

        # # # output
        # drawObject = ImageDraw.Draw(img)
        # drawObject.line([points[0][0]+2,points[0][1]+2,points[1][0]-2,points[1][1]+2],fill="red",width=5)
        # drawObject.line([points[1][0]-2,points[1][1]+2,points[2][0]-2,points[2][1]-2],fill="yellow",width=5)
        # drawObject.line([points[2][0]-2,points[2][1]-2,points[3][0]+2,points[3][1]-2],fill="blue",width=5)
        # drawObject.line([points[3][0]+2,points[3][1]-2,points[0][0]+2,points[0][1]+2],fill="black",width=5)
        # img.save('tmp/{0}.jpg'.format(uuid.uuid1()))

        return (img, points)

    def rotate_points(self, x1, y1, x2, y2, fore_img_w, fore_img_h, angle):
        x1 = x1
        y1 = fore_img_h - y1
        x2 = x2
        y2 = fore_img_h - y2

        x = (x1 - x2)*math.cos(math.pi / 180.0 * angle) - \
            (y1 - y2)*math.sin(math.pi / 180.0 * angle) + x2
        y = (x1 - x2)*math.sin(math.pi / 180.0 * angle) + \
            (y1 - y2)*math.cos(math.pi / 180.0 * angle) + y2
        x = x
        y = fore_img_h - y
        return x, y
