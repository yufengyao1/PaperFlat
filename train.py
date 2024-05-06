'''
Author: yufengyao yufegnyao1@gmail.com
Date: 2023-02-21 13:41:25
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2024-02-04 10:12:23
FilePath: \图像平整\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tqdm import tqdm
import torch
import numpy as np
from loss import PFLDLoss
from pfld import PFLDInference
from datasets import WLFWDatasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    # writer = SummaryWriter('log') #log为tensorboard数据保存路径
    batch_size = 32
    learning_rate=1e-6
    resume=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PFLDInference()
    model_name="pfld"
    model.load_state_dict(torch.load('weights/pfld_29.pth', map_location='cuda')) #196

    model = model.to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)
    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate, weight_decay=1e-6)
    # optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=learning_rate, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # momentum收敛快
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
    if resume:
        optimizer.load_state_dict(torch.load('weights/optimizer.pth', map_location='cpu'))

    transform = transforms.Compose([
        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,hue=0.1),
        # transforms.RandomGrayscale(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.1,scale=(0.02,0.12),value='random'),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = WLFWDatasets('data/train', transform,mode='train',model_type="pfld")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)

    val_dataset = WLFWDatasets('data/val', transform_val,mode='val',model_type="pfld")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model.train()
    for epoch in range(0,2000):
        train_loss, val_loss = 0, 0
        l3_losses = []
        l2_losses=[]
        max_losses=[]
        type_losses=[]
        direction_losses=[]
        for img, label in tqdm(train_loader,leave=False):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            l2_loss,l3_loss,max_loss = criterion(label, out)
            optimizer.zero_grad()
            l2_loss.backward()
            optimizer.step()
            l2_losses.append(l2_loss.item())
            l3_losses.append(l3_loss.item())
            max_losses.append(max_loss.item())
            # type_losses.append(type_loss.item())
            # direction_losses.append(direction_loss.item())

        train_loss = 1000*1.4*np.mean(l3_losses)/8
        train_max_loss = 1000*1.4*np.mean(max_losses)
        # train_type_loss=np.mean(type_losses)
        # train_direction_loss=np.mean(direction_losses)
        train_type_loss=1
        train_direction_loss=1
        
        # model.eval()
        # val_losses=[]
        # with torch.no_grad():
        #     for img, label in val_loader:
        #         img = img.to(device)
        #         label = label.to(device)
        #         out = model(img)
        #         l2_loss,l3_loss,max_loss = criterion(label, out)
        #         val_losses.append(l3_loss.item())
        # val_loss = 1400*np.mean(val_losses)/8
        val_loss=1
        torch.save(model.state_dict(), "weights/{0}_{1}.pth".format(model_name,epoch))
        # torch.save(optimizer.state_dict(),"weights/optimizer.pth")
        print('epoch:{}, train mean loss:{:.6f}, train max loss:{:.6f},val loss:{:.6f},type loss:{:.6f},direction loss:{:.6f}'.format(epoch, train_loss,train_max_loss,val_loss,train_type_loss,train_direction_loss))
