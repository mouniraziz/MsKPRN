import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FeatureEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def l2normalize(x):
    return F.normalize(x, p=2, dim=1)

class KronRelationNets(nn.Module):
    def __init__(self):
        super(KronRelationNets, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(64,64,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*4,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64,8)
        self.fc2 = nn.Linear(8,1)

        self.global_pooling = nn.AdaptiveAvgPool2d([10,10])

    def kron_matching(self, *inputs):
        assert len(inputs) == 2
        assert inputs[0].dim() == 4 and inputs[1].dim() == 4
        assert inputs[0].size() == inputs[1].size()
        N, C, H, W = inputs[0].size()

        w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
        x = inputs[1].view(1, N * C, H, W)
        x = F.conv2d(x, w, groups=N)
        x = x.view(N, H, W, H, W)
        return x

    def forward(self, query, support):
        x = self.global_pooling(query)
        y = self.global_pooling(support)
        b,c,h,w= x.size()
        x = l2normalize(x.view(b,c,h*w)).view(b,c,h,w)
        y = l2normalize(y.view(b,c,h*w)).view(b,c,h,w)
        kron_feature = self.kron_matching(x,y).view(b,h*w, h*w).max(2)[0].view(b,1, h,w).repeat(1,64,1,1)
        kron_feature_ = self.kron_matching(y,x).view(b,h*w, h*w).max(2)[0].view(b,1, h,w).repeat(1,64,1,1)
        x_feature = self.layer0(x)
        y_feature = self.layer0(y)
        out = torch.cat((x_feature, kron_feature, kron_feature_, y_feature),1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out)).view(-1,5)
        return out
