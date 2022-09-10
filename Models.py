import torch
import torch.nn as nn
import numba
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torchvision.models import resnet18
from pointnet.model import PointNetfeat

from Modules import *


def radian2vec(x):
    a, b = torch.cos(x), torch.sin(x)
    return torch.concat((a, b), dim=1)

def vec2radian(x):
    return torch.atan2(x[:, 1:], x[:, :1])

class PointNet(nn.Module):
    def __init__(self, args):

        super(PointNet, self).__init__()

        self.opt = args

        self.net = PointNetfeat()
        self.linear = nn.Linear(1024, args.out_dim)
        

    def forward(self, x):
        x = self.net(x)
        x = self.linear(x[0])

        return x

class PointNetV2(nn.Module):

    def __init__(self, args):

        super(PointNetV2, self).__init__()

        self.opt = args
        self.criterion = nn.MSELoss()

        self.net = PointNetfeat()
        
        self.scaleNet = nn.Linear(1024, 3)
        self.rotNet = nn.Linear(1024, 2)
        self.locNet = nn.Linear(1024, 3)
        

    def forward(self, x):
        x = self.net(x)

        scale = self.scaleNet(x[0])
        rot = self.rotNet(x[0])
        loc = self.locNet(x[0])

        return scale, rot, loc

    def step(self, x, output):
        scale, rot, loc = self.forward(x)

        L1 = self.criterion(scale, output[:, :3])
        L2 = torch.cosine_similarity(rot, radian2vec(output[:, 3: 4])).mean()
        L3 = self.criterion(loc, output[:, 4:])

        # print (rot)
        # print (L2)
        rot = vec2radian(rot)
        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret
    

class PointPillar(nn.Module):
    
    def __init__(self, args):

        super(PointPillar, self).__init__()

        self.opt = args
        self.voxel_gen = VoxelGenerator(args.voxel_size, args.point_cloud_range, args.max_num_points_voxel)

        self.net = resnet18()

    def forward(self, pcd):
        voxels = self.voxel_gen.generate(np.array(pcd[0]))

        return voxels
        
class ManualFeature(nn.Module):

    def __init__(self, args):
        
        super(ManualFeature, self).__init__()

        pcd_range = np.array(args.point_cloud_range)
        self.max_dis = 15

        self.grid_size = (pcd_range[3:] - pcd_range[:3]) // args.voxel_size + 1

        self.locs = torch.zeros(tuple(self.grid_size) + (3,))
        
        low_bound = torch.FloatTensor(pcd_range[:3])
        voxel_size = torch.FloatTensor(args.voxel_size)

        for a in range(self.grid_size[0]):
            for b in range(self.grid_size[1]):
                for c in range(self.grid_size[2]):
                    displacement = torch.FloatTensor([a, b, c]) * voxel_size
                    self.locs[a, b, c] = low_bound +  displacement

        self.locs = self.locs.reshape(-1, 3)
        self.sol = None
        self.criterion = nn.MSELoss()

    def extract(self, pcd):
        N = pcd.size(1)
        self.locs = self.locs.to(pcd.device)
        dis = torch.zeros(self.locs.size(0), pcd.size(0), pcd.size(1)).to(pcd.device)
        feature = torch.zeros(self.max_dis, self.locs.size(0), pcd.size(0)).to(pcd.device)
        Mx = torch.zeros(2, 2).to(pcd.device)
        for i in range(self.locs.size(0)):
            dis[i] = ((pcd - self.locs[i]) ** 2).sum(dim=-1).sqrt().ceil()

        for i in range(self.max_dis):
            feature[i] = (dis <= (i + 1)).sum(dim=-1)
        feature = feature.permute(2, 1, 0)

        return feature
    
    def load(self, path):
        self.sol = torch.tensor(np.load(path))
        # self.sol = torch.load(path)

    def step(self, x, output):
        x = x.permute(0, 2, 1)
        feature = self.extract(x).reshape(x.size(0), -1) / 5000
        sol = self.sol.to(x.device)

        ret = feature @ sol
        # for i in range(7):
        #     ret[i + 1] = ret[i]
        # print (ret[0])
        
        L1 = self.criterion(ret[:, :3], output[:, :3])
        L2 = self.criterion(ret[:, 3], output[:, 3])
        L3 = self.criterion(ret[:, 4:], output[:, 4:])
        # L2, L3 = torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)

        return (L1, L2, L3), ret

class ManualFeature_Net(ManualFeature):

    
    def load(self, path=None):
        self.sol = ConvNet()
        if path is not None:
            self.sol.load_state_dict(torch.load(path))

    def step(self, x, output):
        x = x.permute(0, 2, 1)

        if len(x) <= 16:
            feature = self.extract(x).reshape(x.size(0), -1) / 5000
        else:
            STEP = 8
            feature = None
            for i in range(0, len(x), STEP):
                feat = self.extract(x[i: min(i + STEP, len(x))]).reshape(STEP, -1) / 5000
                if feature is None:
                    feature = feat
                else:
                    feature = torch.cat((feature, feat), dim=0)

        feature = feature.reshape(
            len(x), self.grid_size[0], self.grid_size[1], self.grid_size[2], -1
            ).permute(0, 4, 1, 2, 3)
            
        sol = self.sol.to(x.device)

        return sol.step(feature, output)

class ManualFeature_rot(nn.Module):

    def __init__(self, args):
        
        super(ManualFeature_rot, self).__init__()

        pcd_range = np.array(args.point_cloud_range)
        self.max_dis = 15

        self.grid_size = (pcd_range[3:] - pcd_range[:3]) // args.voxel_size + 1
        self.voxel_size = torch.FloatTensor(args.voxel_size)
        
        self.locs = torch.zeros(tuple(self.grid_size) + (3,))

        angs = torch.FloatTensor([(np.pi / args.ang_bins * ang - np.pi / 2) for ang in range(args.ang_bins)])
        mats = []

        for ang in angs:
            mats.append([
                [torch.cos(ang), -torch.sin(ang), 0],
                [torch.sin(ang), torch.cos(ang), 0],
                [0, 0, 1]
            ])

        self.mats = torch.FloatTensor(mats)
        
        low_bound = torch.FloatTensor(pcd_range[:3])
        voxel_size = torch.FloatTensor(args.voxel_size)

        self.offset = -low_bound + torch.FloatTensor(args.voxel_size) / 2

        for a in range(self.grid_size[0]):
            for b in range(self.grid_size[1]):
                for c in range(self.grid_size[2]):
                    displacement = torch.FloatTensor([a, b, c]) * voxel_size
                    self.locs[a, b, c] = low_bound +  displacement

        self.locs = self.locs.reshape(-1, 3)
        self.sol = None
        self.criterion = nn.MSELoss()

    def extract(self, pcd):
        N = pcd.size(1)

        self.locs = self.locs.to(pcd.device)
        self.mats = self.mats.to(pcd.device)
        self.offset = self.offset.to(pcd.device)
        self.voxel_size = self.voxel_size.to(pcd.device)

        dis = torch.zeros(self.locs.size(0), pcd.size(0), pcd.size(1)).to(pcd.device)
        feature = torch.zeros(self.max_dis, self.locs.size(0), pcd.size(0)).to(pcd.device)
        feature_rot = torch.zeros(self.mats.size(0), pcd.size(0), self.locs.size(0)).to(pcd.device)

        for i in range(self.locs.size(0)):
            dis[i] = ((pcd - self.locs[i]) ** 2).sum(dim=-1).sqrt().ceil()

        for i in range(self.max_dis):
            feature[i] = (dis <= (i + 1)).sum(dim=-1)
        feature = feature.permute(2, 1, 0)

        # for i in range(10):
        #     pts = np.array(pcd[i].to('cpu'))
        #     print (pts.shape)
        #     np.save(f'/projects/perception/personals/wenjiey/works/tools/imgs/{i + 1}/pts.npy', pts)
        
        # quit()

        for i in range(len(self.mats)):
            pts = pcd @ self.mats[i].T + self.offset
            pts = (pts / self.voxel_size).floor()

            pts = pts[..., 2] + pts[..., 1] * self.grid_size[2] + pts[..., 0] * self.grid_size[1] * self.grid_size[2]

            pts = pts.long()
            for j in range(pcd.size(0)):
                feature_rot[i, j] = torch.histc(pts[j], bins=self.locs.size(0), min=-0.5, max=self.locs.size(0) - 0.5)
            
            # for j in range(pcd.size(0)):
            #     for pt in pts[j]:
            #         if pt >= 0 and pt < self.locs.size(0):
            #             feature_rot[i, j, pt] += 1
        # quit()
        feature_rot = feature_rot.permute(1, 2, 0)
        feature = torch.cat((feature, feature_rot), axis=-1) / pcd.size(1)

        return feature
    
    def load(self, path):
        self.sol = torch.tensor(np.load(path))
        # self.sol = torch.load(path)

    def step(self, x, output):
        x = x.permute(0, 2, 1)
        feature = self.extract(x).reshape(x.size(0), -1)
        sol = self.sol.to(x.device)

        ret = feature @ sol
        # for i in range(7):
        #     ret[i + 1] = ret[i]
        # print (ret[0])
        
        L1 = self.criterion(ret[:, :3], output[:, :3])
        L2 = self.criterion(ret[:, 3], output[:, 3])
        L3 = self.criterion(ret[:, 4:], output[:, 4:])
        # L2, L3 = torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)

        return (L1, L2, L3), ret

# pts = torch.rand(100, 3)
# angs = torch.FloatTensor([0, np.pi / 2])
# mats = vec2radian(angs)