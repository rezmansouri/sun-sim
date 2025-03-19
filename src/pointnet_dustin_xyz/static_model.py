from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            4, 64, 1
        )  # Input channels remain 4 (x, y, z, intensity)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))  # Removed transformation on x

        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        print(x.shape, point_feat.shape)

        if self.global_feat:
            return x  # No transformation matrix returned
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], 1)


class PointNetModel(nn.Module):
    def __init__(self):
        super(PointNetModel, self).__init__()
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.sigmoid(x)
        x = x.view(batchsize, n_pts, 1)
        return x


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 4, 10))

    seg = PointNetModel()
    out = seg(sim_data)
    print("seg", out.size())
