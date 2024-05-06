'''
Author: yufengyao yufegnyao1@gmail.com
Date: 2023-02-21 13:41:25
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2023-12-06 13:54:22
FilePath: \图像平整\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tqdm import tqdm
import torch
import numpy as np
from loss import PFLDLoss
# from pfld import PFLDInference
from pfld import PFLDInference
from datasets import WLFWDatasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from mobilevit import mobilevit_xxs,mobilevit_xs,mobilevit_s
if __name__ == '__main__':
    # writer = SummaryWriter('log') #log为tensorboard数据保存路径

    batch_size = 32
    learning_rate=1e-6
    resume=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = mobilevit_xs()
    model_name="mobilevitxs"
    model.load_state_dict(torch.load('weights/mobilevitxs_192.pth', map_location='cuda')) #196

    model = model.to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate, weight_decay=1e-6)
    if resume:
        optimizer.load_state_dict(torch.load('weights/optimizer_mobilevitxs.pth', map_location='cpu'))
    
    # optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.05)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # momentum收敛快
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)

    transform = transforms.Compose([
        transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,hue=0.1),
        # transforms.RandomGrayscale(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.1,scale=(0.02,0.12),value='random'),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2,hue=0.1),
        # transforms.RandomGrayscale(p=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.05,scale=(0.02,0.12),value='random'),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = WLFWDatasets('data/train', transform,mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)

    val_dataset = WLFWDatasets('data/val', transform_val,mode='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model.train()
    for epoch in range(0,2000):
        # model.train()
        train_loss, val_loss = 0, 0
        losses = []
        max_losses=[]
        for img, label in tqdm(train_loader,leave=False):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss,l3_loss,max_loss = criterion(label, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(l3_loss.item())
            max_losses.append(max_loss.item())
        # scheduler.step()
        train_loss = 1000*1.4*np.mean(losses)/8
        train_max_loss = 1000*1.4*np.mean(max_losses)
        
        # model.eval()
        # losses.clear()
        # with torch.no_grad():
        #     for img, label in val_loader:
        #         img = img.to(device)
        #         label = label.to(device)
        #         out = model(img)
        #         loss,l2_loss,max_loss = criterion(label, out)
        #         # loss = torch.mean(torch.sum((label - out)**2, axis=1))
        #         losses.append(l2_loss.item())
        # val_loss = 1400*np.mean(losses)/8
        val_loss=1
        # writer.add_scalar("train loss",train_loss,epoch)
        torch.save(model.state_dict(), "weights/{0}_{1}.pth".format(model_name,epoch))
        if resume:
            torch.save(optimizer.state_dict(),"weights/optimizer_mobilevitxs.pth")
        print('epoch:{}, train mean loss:{:.6f}, train max loss:{:.6f},val loss:{:.6f}'.format(epoch, train_loss,train_max_loss,val_loss))
