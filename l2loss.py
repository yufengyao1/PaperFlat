import torch
from torch import nn

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()
        self.crossentropyloss = nn.CrossEntropyLoss()

    def forward(self, label, out):
        box_label=label[:, :8]
        box_out=out[:, :8]
        
        type_label=label[:, 8:10]
        type_out=out[:, 8:10]
        
        direction_label=label[:,10:]
        direction_out=out[:,10:]
        
        direction_loss=self.crossentropyloss(direction_out,direction_label)
        type_loss=self.crossentropyloss(type_out,type_label)
        
        l2_loss = torch.mean(torch.sum((box_label- box_out) * (box_label - box_out), axis=1))
        
        # l3_loss = torch.mean(torch.sum(torch.abs(box_label - box_out) *type_label[:,0:1], axis=1)) #四个点的距离差加和
        # max_loss = torch.mean(torch.max(torch.abs(box_label - box_out) *type_label[:,0:1], axis=1)[0]) #四个点的距离差加和
        l3_loss = torch.mean(torch.sum(torch.abs(box_label - box_out) , axis=1)) #四个点的距离差加和
        max_loss = torch.mean(torch.max(torch.abs(box_label - box_out) , axis=1)[0]) #四个点的距离差加和
  
        all_loss = l2_loss+0.01*type_loss +direction_loss
        return all_loss, l3_loss, max_loss,type_loss,direction_loss

   
