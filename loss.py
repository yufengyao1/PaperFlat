'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-13 11:27:36
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2024-01-09 14:30:03
FilePath: \PFLD-MINE\loss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, label, out):
        
        l2_loss = torch.mean(torch.sum((label[:, :8] - out[:, :-1]) * (label[:, :8] - out[:, :-1])*label[:, 8:9], axis=1))
        l3_loss = torch.mean(torch.sum(torch.abs(label[:, :8] - out[:, :-1]) *label[:, 8:9], axis=1)) #四个点的距离差加和
        max_loss = torch.mean(torch.max(torch.abs(label[:, :8] - out[:, :-1]) *label[:, 8:9], axis=1)[0]) #四个点的距离差加和
        type_loss = torch.mean(torch.sum((label[:, 8:9] - out[:, -1:]) ** 2, axis=1))
        # l2_loss=self.wing_loss(label,out)
        
        all_loss = 0.99*l2_loss+0.01*type_loss
        return all_loss, l3_loss, max_loss

    # def forward(self, label, out):
    #     l3_loss = torch.mean(torch.sum(torch.abs(label[:, :-1] - out[:, :-1]) * label[:, -1:], axis=1))  # 四个点的距离差加和
    #     type_loss = torch.sum((label[:, -1:] - out[:, -1:]) ** 2, axis=1)
    #     l2_loss = torch.sum((label[:, :-1] - out[:, :-1]) ** 2, axis=1)
    #     all_loss = torch.mean(type_loss * l2_loss)
    #     # print(l2_loss)Ï
    #     return all_loss, l3_loss