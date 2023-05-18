import torch
from torch import nn
import numpy as np
from torch.nn import CrossEntropyLoss

def onehot(x,n):
    retval = np.zeros(n)
    if x > 0:
        retval[x-1] = 1
    return retval

class PlanLoss(nn.Module):
    def __init__(self):
        super(PlanLoss, self).__init__()
    def getWeights(self, output, target):
        # return 1
        f1 = (1+5*torch.stack([2-torch.sum(target.reshape(-1,21,21),dim=-1)]*21,dim=-1)).reshape(-1,21*21)
        f2 = 100*target + 1
        return (f1+f2)/60
        exit(0)
        # print(max(torch.sum(target.reshape(21,21),dim=-1)))
        return (target*torch.sum(target,dim=-1) + 1)
    def MSELoss(self, output, target):
        retval  = (output - target)**2
        retval *= self.getWeights(output,target)
        return torch.mean(retval)
    def BCELoss(self, output, target, loss_mask=None):
        mask_factor = torch.ones(target.shape).to(output.device)
        if loss_mask is not None:
            loss_mask = loss_mask.reshape(-1,21,21)
            mask_factor = mask_factor.reshape(-1,21,21)
            # print(mask_factor.shape,loss_mask.shape,output.shape,target.shape)
            for idx, tgt in enumerate(loss_mask):
                for jdx, tgt_node in enumerate(tgt):
                    if sum(tgt_node) == 0:
                        mask_factor[idx,jdx] *= 0
            
        # print(loss_mask[0].data.cpu().numpy())
        # print(mask_factor[0].data.cpu().numpy())
        # print()
        # print(loss_mask[45].data.cpu().numpy())
        # print(mask_factor[45].data.cpu().numpy())
        # print()
        # print(loss_mask[-1].data.cpu().numpy())
        # print(mask_factor[-1].data.cpu().numpy())
        # print()


        loss_mask = loss_mask.reshape(-1,21*21)
        mask_factor = mask_factor.reshape(-1,21*21)
        # print(loss_mask.shape, target.shape, mask_factor.shape)
        # exit()

        factor = (10 if target.shape[-1]==441 else 1)# * torch.sum(target,dim=-1)+1
        retval  = -1 * mask_factor * (factor * target * torch.log(1e-6+output) + (1-target) * torch.log(1e-6+1-output))
        factor = torch.stack([torch.sum(target,dim=-1)+1]*target.shape[-1],dim=-1)
        return torch.mean(factor*retval)
        return torch.mean(retval)
    def forward(self, output, target, loss_mask=None):
        return self.BCELoss(output,target,loss_mask) + 0.01*torch.sum(output - 1/21)
        # return self.MSELoss(output,target)
        
class DialogueActLoss(nn.Module):
    def __init__(self):
        super(DialogueActLoss, self).__init__()
        self.bias  = torch.tensor([289,51,45,57,14,12,1,113,6,264,27,63,22,66,2,761,129,163,5]).float()
        self.bias  = max(self.bias) - self.bias + 1
        self.bias /= torch.sum(self.bias)
        self.bias  = 1-self.bias
        # self.bias *= self.bias
    def BCELoss(self, output, target):
        target = torch.stack([torch.tensor(onehot(x + 1,19)).long() for x in target]).to(output.device)
        retval  = -1 * (target * torch.log(1e-6+output) + (1-target) * torch.log(1e-6+1-output))
        retval *= torch.stack([self.bias] * output.shape[0]).to(output.device)
        # print(output)
        # print(target)
        # print(retval)
        # print(torch.mean(retval))
        # exit()
        return torch.mean(retval)
    def forward(self, output, target):
        return self.BCELoss(output,target)
        # return self.MSELoss(output,target)
        
class DialogueMoveLoss(nn.Module):
    def __init__(self, device):
        super(DialogueMoveLoss, self).__init__()
        # self.bias  = torch.tensor([289,51,45,57,14,12,1,113,6,264,27,63,22,66,2,761,129,163,5]).float()
        # self.bias  = max(self.bias) - self.bias + 1
        # self.bias /= torch.sum(self.bias)
        # self.bias  = 1-self.bias
        move_weights = torch.tensor(np.array([202, 34, 34, 48, 4, 2, 420, 10, 54, 1, 10, 11, 30, 28, 14, 2, 16, 6, 2, 86, 4, 12, 28, 2, 2, 16, 12, 14, 4, 1, 12, 258, 12, 26, 2])).float().to(device)
        move_weights = 1+ max(move_weights) - move_weights
        self.loss1 = CrossEntropyLoss(weight=move_weights)
        zero_bias = 0.773
        num_classes = 40

        weight = torch.tensor(np.array([50 if not x else 1 for x in range(num_classes)])).float().to(device)
        weight = 1+ max(weight) - weight
        self.loss2 = CrossEntropyLoss(weight=weight)
        # self.bias *= self.bias
    def BCELoss(self, output, target,zero_bias):
        # # print(output.shape,target.shape)
        # bias = torch.tensor(np.array([1 if t else zero_bias for t in target])).to(output.device)
        # target = torch.stack([torch.tensor(onehot(x,output.shape[-1])).long() for x in target]).to(output.device)
        # # print(target.shape, bias.shape, bias)
        
        # retval  = -1 * (target * torch.log(1e-6+output) + (1-target) * torch.log(1e-6+1-output))
        # retval = torch.mean(retval,-1)
        
        # # print(retval.shape)
        # retval *= bias
        # # retval *= torch.stack([self.bias] * output.shape[0]).to(output.device)
        # # print(output)
        # # print(target)
        # # print(retval)
        # # print(torch.mean(retval))
        # # exit()
        # # retval = self.loss(output,target)
        # return torch.mean(retval) # retval #
        # weight = [zero_bias if x else (1-zero_bias)/(output.shape[-1]-1) for x in range(output.shape[-1])]
        retval = self.loss2(output,target) if zero_bias else self.loss1(output,target)
        return retval #
    def forward(self, output, target):
        o1, o2, o3, o4 = output
        t1, t2, t3, t4 = target

        # print(t2,t2.shape, o2.shape)

        # if sum(t2):
        #     o2, t2 = zip(*[(a,b) for a,b in zip(o2,t2) if b])
        #     o2 = torch.stack(o2)
        #     t2 = torch.stack(t2)
        # if sum(t3):
        #     o3, t3 = zip(*[(a,b) for a,b in zip(o3,t3) if b])
        #     o3 = torch.stack(o3)
        #     t3 = torch.stack(t3)
        # if sum(t4):
        #     o4, t4 = zip(*[(a,b) for a,b in zip(o4,t4) if b])
        #     o4 = torch.stack(o4)
        #     t4 = torch.stack(t4)

        # print(t2,t2.shape, o2.shape)
        # exit()

        retval = sum([
            1*self.BCELoss(output[0],target[0],0),
            0*self.BCELoss(output[1],target[1],1),
            0*self.BCELoss(output[2],target[2],1),
            0*self.BCELoss(output[3],target[3],1)
        ])
        return retval #sum([fact*self.BCELoss(o,t,zbias) for fact,zbias,o,t in zip([1,0,0,0],[0,1,1,1],output,target)])
        # return self.MSELoss(output,target)
        
class DialoguePredLoss(nn.Module):
    def __init__(self):
        super(DialoguePredLoss, self).__init__()
        self.bias  = torch.tensor([289,51,45,57,14,12,1,113,6,264,27,63,22,66,2,761,129,163,5,0]).float()
        self.bias[-1] = 1460#2 * torch.sum(self.bias) // 3
        self.bias  = max(self.bias) - self.bias + 1
        self.bias /= torch.sum(self.bias)
        self.bias  = 1-self.bias
        # self.bias *= self.bias
    def BCELoss(self, output, target):
        target = torch.stack([torch.tensor(onehot(x + 1,20)).long() for x in target]).to(output.device)
        retval  = -1 * (target * torch.log(1e-6+output) + (1-target) * torch.log(1e-6+1-output))
        retval *= torch.stack([self.bias] * output.shape[0]).to(output.device)
        # print(output)
        # print(target)
        # print(retval)
        # print(torch.mean(retval))
        # exit()
        return torch.mean(retval)
    def forward(self, output, target):
        return self.BCELoss(output,target) 
        # return self.MSELoss(output,target)
        
class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bias1  = torch.tensor([134,1370,154,128,220,166,46,76,106,78,88,124,102,120,276,122,112,106,44,174,20]).float()
        # self.bias[-1] = 1460#2 * torch.sum(self.bias) // 3
        # self.bias1  = torch.ones(21).float()
        self.bias1  = max(self.bias1) - self.bias1 + 1
        self.bias1 /= torch.sum(self.bias1)
        self.bias1  = 1-self.bias1
        self.bias2  = torch.tensor([1168,1310]).float()
        # self.bias2[-1] = 1460#2 * torch.sum(self.bias) // 3
        # self.bias2  = torch.ones(21).float()
        self.bias2  = max(self.bias2) - self.bias2 + 1
        self.bias2 /= torch.sum(self.bias2)
        self.bias2  = 1-self.bias2
        # self.bias *= self.bias
    def BCELoss(self, output, target):
        # target = torch.stack([torch.tensor(onehot(x + 1,20)).long() for x in target]).to(output.device)
        retval  = -1 * (target * torch.log(1e-6+output) + (1-target) * torch.log(1e-6+1-output))
        # print(self.bias1.shape,self.bias2.shape,output.shape[-1])
        retval *= torch.stack([self.bias2 if output.shape[-1]==2 else self.bias1] * output.shape[0]).to(output.device)
        # print(output)
        # print(target)
        # print(retval)
        # print(torch.mean(retval))
        # exit()
        return torch.mean(retval)
    def forward(self, output, target):
        return self.BCELoss(output,target) 
        # return self.MSELoss(output,target)