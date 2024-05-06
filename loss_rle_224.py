import math
import torch
import numpy as np
from torch import nn
from realnvp import RealNVP
from torch import distributions


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())
def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


class RLELoss(nn.Module):
    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis
        self.flow_model = RealNVP()
        self.flow_model.to("cuda")

    def forward(self, output, target, type_label,target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        pred = output[:, :, :2]
        sigma = output[:, :, 2:4].sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        # error[error > 1e+5] = 0
        # error[error < -1e+5] = 0
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], -1,2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss
        # loss[loss > 1e+5] = 0
        # loss[loss < -1e+5] = 0

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight
            
        # loss=loss.view(loss.size(0),-1)
        # loss*=type_label[:,0:1] #去除无前景
        if self.size_average:
            loss /= len(loss)
        return loss.sum()

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()
        self.crossentropyloss = nn.CrossEntropyLoss()
        self.rleloss=RLELoss()

    def forward(self, label, out):
        box_label=label[:, :8]
        box_out=out[:, :16]
        type_label=label[:, 8:10]
        type_out=out[:, 16:18]
        
        # direction_label=label[:,10:]
        # direction_out=out[:,18:]
        
        #direction_loss=self.crossentropyloss(direction_out,direction_label)
        direction_loss=torch.tensor([1])
        type_loss=self.crossentropyloss(type_out,type_label)
        

        box_label = box_label.view(box_label.size(0), 4,2)
        box_out = box_out.view(box_out.size(0), 4,4)
        l2_loss=self.rleloss(box_out,box_label,type_label)
        
        box_out=box_out[:,:,:2]
        
        box_out=box_out.reshape(box_out.size(0),-1)
        box_label=box_label.reshape(box_label.size(0),-1)
        
        # l3_loss = torch.mean(torch.sum(torch.abs(box_label - box_out) *type_label[:,0:1], axis=1)) #四个点的距离差加和
        # max_loss = torch.mean(torch.max(torch.abs(box_label - box_out) *type_label[:,0:1], axis=1)[0]) #四个点的距离差加和
        l3_loss = torch.mean(torch.sum(torch.abs(box_label - box_out) , axis=1)) #四个点的距离差加和
        max_loss = torch.mean(torch.max(torch.abs(box_label - box_out) , axis=1)[0]) #四个点的距离差加和
  
        all_loss = 0.99*l2_loss+0.01*type_loss# +50*direction_loss
        # all_loss = l2_loss
        
        return all_loss, l3_loss, max_loss,type_loss,direction_loss

   
