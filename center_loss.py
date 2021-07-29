#######################
#author: Shiming Chen
#FREE
#######################

import torch
import torch.nn as nn

def Other_label(labels,num_classes):
    index=torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    other_labels=labels+index
    other_labels[other_labels >= num_classes]=other_labels[other_labels >= num_classes]-num_classes
    return other_labels


class TripCenterLoss_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=312, use_gpu=True):
        super(TripCenterLoss_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim)) 
    def forward(self, x, labels,margin, incenter_weight):
        other_labels = Other_label(labels, self.num_classes)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]
        other_labels = other_labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask_other = other_labels.eq(classes.expand(batch_size, self.num_classes))
        dist_other = distmat[mask_other]
        loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_other,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

class TripCenterLoss_min_margin(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(TripCenterLoss_min_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels,margin, incenter_weight):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other=torch.FloatTensor(batch_size,self.num_classes-1).cuda()
        for i in range(batch_size):
            other[i]=(distmat[i,mask[i,:]==0])

        dist_min,_=other.min(dim=1)
        loss = torch.max(margin+incenter_weight*dist-(1-incenter_weight)*dist_min,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss