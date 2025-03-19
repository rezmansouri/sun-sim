from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, dims=[64, 128, 1024], global_feat=True):
        super(PointNetfeat, self).__init__()
        convs, bns = [nn.Conv1d(4, dims[0], 1)], []
        for i in range(len(dims) - 1):
            convs.append(nn.Conv1d(dims[i], dims[i + 1], 1))
            bns.append(nn.BatchNorm1d(dims[i]))
        bns.append(nn.BatchNorm1d(dims[-1]))
        self.convs, self.bns, self.dims, self.depth = (
            nn.ModuleList(convs),
            nn.ModuleList(bns),
            dims,
            len(dims),
        )
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bns[0](self.convs[0](x)))  # Removed transformation on x
        point_feat = x
        for i in range(1, self.depth - 1):
            x = F.relu(self.bns[i](self.convs[i](x)))
        x = self.bns[-1](self.convs[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.dims[-1])

        if self.global_feat:
            return x  # No transformation matrix returned
        else:
            x = x.view(-1, self.dims[-1], 1).repeat(1, 1, n_pts)
            return torch.cat([x, point_feat], 1)


class PointNetModel(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512, 1024]):
        super(PointNetModel, self).__init__()
        self.feat = PointNetfeat(dims=dims, global_feat=False)
        self.dims = dims[::-1]
        convs, bns = [nn.Conv1d(self.dims[0] + self.dims[-1], self.dims[1], 1)], []
        for i in range(1, len(self.dims) - 1):
            convs.append(nn.Conv1d(self.dims[i], self.dims[i + 1], 1))
            bns.append(nn.BatchNorm1d(self.dims[i]))
        bns.append(nn.BatchNorm1d(self.dims[-1]))
        convs.append(nn.Conv1d(self.dims[-1], 1, 1))
        self.convs, self.bns, self.depth = (
            nn.ModuleList(convs),
            nn.ModuleList(bns),
            len(self.dims),
        )

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        for i in range(self.depth - 1):
            x = F.relu(self.bns[i](self.convs[i](x)))
        x = self.convs[-1](x)
        x = x.transpose(2, 1).contiguous()
        # x = F.sigmoid(x)
        x = x.view(batchsize, n_pts, 1)
        return x


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 4, 10))

    seg = PointNetModel(dims=[64, 128, 256, 512, 1024, 2048, 4096])
    out = seg(sim_data)
    print("seg", out.size())

    # pointnetfeat = PointNetfeat(dims=[64, 128, 256, 512, 1024])
    # print(pointnetfeat(sim_data).shape)
