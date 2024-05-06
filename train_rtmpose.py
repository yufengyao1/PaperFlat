'''
Author: yufengyao yufegnyao1@gmail.com
Date: 2024-01-19 11:10:31
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2024-02-01 21:33:42
FilePath: \图像平整\train_rtmpose.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import numpy as np
from tqdm import tqdm
from datasets import WLFWDatasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.models.backbones.cspnext import CSPNeXt
from mmpose.models.losses.classification_loss import KLDiscretLoss
if __name__ == '__main__':
    batch_size = 64
    learning_rate=1e-6
    resume=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CSPNeXt()
    model_name="rtmpose"
    model.load_state_dict(torch.load('weights/rtmpose_38.pth', map_location='cuda')) #196

    model = model.to(device)
    criterion = KLDiscretLoss()
    
    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate, weight_decay=0)
    # optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=learning_rate, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # momentum收敛快
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.9)
    if resume:
        optimizer.load_state_dict(torch.load('weights/optimizer.pth', map_location='cpu'))

    transform = transforms.Compose([
        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.1, contrast=0.2,hue=0.1),
        transforms.RandomGrayscale(p=0.5),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_dataset = WLFWDatasets('data/train', transform,mode='train',model_type="rtmpose")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)

    simcc_label=SimCCLabel(input_size=(224, 224),sigma=(4.9, 5.66),simcc_split_ratio=2.0,normalize=False,use_dark=False)
    model.train()
    for epoch in range(0,2000):
        train_loss, val_loss = 0, 0
        loss_list=[]
        for img, label in tqdm(train_loader,leave=False):
            img = img.to(device)
            pred = model(img)
            
            label=label[:,:8]
            label=label.view(batch_size,4,2)
            label=label.numpy()
            label=simcc_label.encode(label)
            
            label=list(label.values())
            target_weights=label[2]
            target=torch.from_numpy(np.array(label[0:2],dtype=np.float32))
            target_weights=torch.from_numpy(np.array(target_weights,dtype=np.float32))
            
            target = Variable(target).to(device)
            target_weights = Variable(target_weights).to(device)
            
            loss= criterion(pred,target,target_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        train_loss = np.mean(loss_list)
        train_max_loss=-1
        train_type_loss=-1
        train_direction_loss=-1
        val_loss=1
        
        torch.save(model.state_dict(), "weights/{0}_{1}.pth".format(model_name,epoch))
        # torch.save(optimizer.state_dict(),"weights/optimizer.pth")
        print('epoch:{}, train mean loss:{:.6f}, train max loss:{:.6f},val loss:{:.6f},type loss:{:.6f},direction loss:{:.6f}'.format(epoch, train_loss,train_max_loss,val_loss,train_type_loss,train_direction_loss))
        # scheduler.step()
