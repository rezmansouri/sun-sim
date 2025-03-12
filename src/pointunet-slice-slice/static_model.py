from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torchsummary import summary


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = self.bn5(self.conv5(x4))
        x6 = torch.max(x5, 2, keepdim=True)[0]
        x7 = x6.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x7, x3], 1), [x1, x2, x3, x4]


class PointNetModel(nn.Module):
    def __init__(self):
        super(PointNetModel, self).__init__()
        self.feat = PointNetfeat()
        self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(512, 128, 1)
        self.conv4 = torch.nn.Conv1d(256, 64, 1)
        self.conv5 = torch.nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, [xx1, xx2, xx3, xx4] = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(torch.cat((x, xx4), dim=1))))
        x = F.relu(self.bn3(self.conv3(torch.cat((x, xx3), dim=1))))
        x = F.relu(self.bn4(self.conv4(torch.cat((x, xx2), dim=1))))
        x = self.conv5(torch.cat((x, xx1), dim=1))
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, 1)
        return x


if __name__ == "__main__":
    sim_data = torch.rand(32, 4, 10)

    seg = PointNetModel()
    out = seg(sim_data)
    print("seg", out.size())

    # encoder = PointNetfeat()
    # summary(encoder, (4, 10))

    # full = PointNetModel()
    # summary(full, (4, 10))
    
    
