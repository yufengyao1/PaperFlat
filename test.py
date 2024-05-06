import os
import cv2
import copy
import time
import json
import torch
import numpy as np
from PIL import Image
from pfld import PFLDInference
from torchvision import transforms
from torch.autograd import Variable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = PFLDInference()
model.load_state_dict(torch.load('weights/pfld_1212.pth', map_location='cpu'))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
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
    if (heightA+heightB) > (widthA+widthB):
        maxHeight = int(maxWidth*4/3)
    else:
        maxHeight = int(maxWidth*3/4)
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), borderValue=(175, 175, 175))
    warped = warped[10:-10, 10:-10]
    return warped


def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img


def white_balance_2(img_input):
    '''
    完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r

    return img


def white_balance_3(img):
    '''
    灰度世界假设
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)

    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


def white_balance_4(img):
    '''
    基于图像分析的偏色检测及颜色校正方法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''

    def detection(img):
        '''计算偏色值'''
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b, M_a, M_b = 0, 0, 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
        D = np.sqrt((np.square(d_a) + np.square(d_b)))

        for i in range(m):
            for j in range(n):
                M_a = np.abs(a[i][j] - d_a - 128) + M_a
                M_b = np.abs(b[i][j] - d_b - 128) + M_b

        M_a, M_b = M_a / (m * n), M_b / (m * n)
        M = np.sqrt((np.square(M_a) + np.square(M_b)))
        k = D / M
        print('偏色值:%f' % k)
        return

    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    # detection(img)

    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(b[0][0]), int(g[0][0])
    for i in range(m):
        for j in range(n):
            I_r_2[i][j] = int(r[i][j] ** 2)
            I_b_2[i][j] = int(b[i][j] ** 2)
            sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
            sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
            sum_I_g = g[i][j] + sum_I_g
            sum_I_r = r[i][j] + sum_I_r
            sum_I_b = b[i][j] + sum_I_b
            if max_I_r < r[i][j]:
                max_I_r = r[i][j]
            if max_I_r_2 < I_r_2[i][j]:
                max_I_r_2 = I_r_2[i][j]
            if max_I_g < g[i][j]:
                max_I_g = g[i][j]
            if max_I_b_2 < I_b_2[i][j]:
                max_I_b_2 = I_b_2[i][j]
            if max_I_b < b[i][j]:
                max_I_b = b[i][j]

    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    # print(u_b, v_b, u_r, v_r)
    b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
            g0[i][j] = g[i][j]
            # r0[i][j] = r[i][j]
            r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
            if r_point > 255:
                r0[i][j] = 255
            else:
                if r_point < 0:
                    r0[i][j] = 0
                else:
                    r0[i][j] = r_point
            if b_point > 255:
                b0[i][j] = 255
            else:
                if b_point < 0:
                    b0[i][j] = 0
                else:
                    b0[i][j] = b_point
    return cv2.merge([b0, g0, r0])


def white_balance_5(img):
    '''
    动态阈值算法
    算法分为两个步骤：白点检测和白点调整。
    只是白点检测不是与完美反射算法相同的认为最亮的点为白点，而是通过另外的规则确定
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''

    b, g, r = cv2.split(img)
    """
    YUV空间
    """
    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]

    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)

    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # 如果该值过大过小，色温向两极端发展
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0

            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.1 * ysum:    # 取前10%的亮点为计算值，如果该值过大易过曝光，该值过小调整幅度小
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb

    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            if b_point > 255:
                b[i][j] = 255
            else:
                if b_point < 0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point > 255:
                g[i][j] = 255
            else:
                if g_point < 0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point > 255:
                r[i][j] = 255
            else:
                if r_point < 0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point

    return cv2.merge([b, g, r])


folder = "data/img_test_bad"
files = os.listdir(folder)
files = [os.path.join(folder, f) for f in files]


for index, file in enumerate(files[0:-1]):
    print(f"分析第 {index} 个")
    frame = cv2.imread(file)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_ori = img_pil.copy()

    img_pil = transform(img_pil)
    inputs = Variable(torch.unsqueeze(img_pil, dim=0).float(), requires_grad=False)
    inputs = inputs.to(device)
    landmark = model(inputs)
    landmark = landmark.tolist()[0]
    score = landmark[-1]
    landmark_origin = copy.deepcopy(landmark[0:8])

    h, w = frame.shape[0], frame.shape[1]
    for i in range(8):
        if i % 2 == 0:
            landmark[i] = landmark[i]*w
        else:
            landmark[i] = landmark[i]*h

    points = []
    for i in range(4):
        points.append((int(landmark[2*i]), int(landmark[2*i+1])))
    
    frame_rect=frame.copy()
    # for pt in points:
    #     cv2.circle(image, pt, 2, (0, 0, 255), cv2.FILLED,lineType=cv2.LINE_AA)
    # cv2.circle(frame_rect, (int(points[0][0]), int(points[0][1])), 40, (255, 0, 0), cv2.FILLED, lineType=cv2.LINE_AA)
    points_rect = (np.array(points)).reshape((-1, 1, 2))
    cv2.polylines(frame_rect, [points_rect], isClosed=True, color=(0, 0, 255), thickness=15, lineType=cv2.LINE_AA)
    cv2.imwrite(f"tmp/{index}.jpg", frame_rect)
    
    frame_transformed = four_point_transform(frame, points)

    frame_white_balance = white_balance_2(frame_transformed)
    cv2.imwrite(f"tmp/{index}_trans.jpg", frame_white_balance)
    continue

    scale = 400/frame.shape[0]
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    h, w = frame.shape[0], frame.shape[1]
    for i in range(8):
        if i % 2 == 0:
            landmark_origin[i] = landmark_origin[i]*w
        else:
            landmark_origin[i] = landmark_origin[i]*h
    points = []
    for i in range(4):
        points.append((int(landmark_origin[2*i]), int(landmark_origin[2*i+1])))
    if score > 0.8:
        cv2.circle(frame, (int(points[0][0]), int(points[0][1])), 5, (255, 0, 0), cv2.FILLED, lineType=cv2.LINE_AA)
        points = (np.array(points)).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    scale = 610/frame_transformed.shape[0]
    frame_transformed = cv2.resize(frame_transformed, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    frame_transformed = frame_transformed[5:-5, 5:-5, :]

    cv2.imshow('paper1', frame)
    cv2.imshow('paper2', frame_transformed)
    print(file)
    k = cv2.waitKey()
    if k == 27:
        break
