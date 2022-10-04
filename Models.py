import torch
import torch.nn as nn
import cv2
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torchvision.models import resnet18
from pointnet.model import PointNetfeat

from Modules import *

from utils import Plot


def radian2vec(x):
    a, b = torch.cos(x), torch.sin(x)
    return torch.concat((a, b), dim=1)

def vec2radian(x):
    return torch.atan2(x[..., 1:], x[..., :1])

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

    
    def load(self, path):
        # self.sol = ConvNet()
        self.sol = ConvNetV2()
        self.sol.load_state_dict(torch.load(path))
        # self.sol = torch.load(path)

    def step(self, x, output):
        x = x.permute(0, 2, 1)
        feature = self.extract(x) / 5000
        feature = feature.reshape(len(feature), 11, 11, 11, -1).permute(0, 4, 1, 2, 3)
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

class ManualFeature_rot_Net(ManualFeature_rot):

    def load(self, path):
        self.sol = ConvNetV2()
        self.sol.load_state_dict(torch.load(path))

    def step(self, x, output):
        x = x.permute(0, 2, 1)
        feature = self.extract(x)
        feature = feature.reshape(len(feature), 11, 11, 11, -1).permute(0, 4, 1, 2, 3)
        sol = self.sol.to(x.device)

        return sol.step(feature, output)

class OPBB(nn.Module):

    def __init__(self):
        super(OPBB, self).__init__()
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x_2d = x[..., :2]
        z_low = x[..., 2].min(-1)[0]
        z_upp = x[..., 2].max(-1)[0]

        rec = []
        ret = torch.zeros(x.size(0), 7)
        ret[..., 2] = z_upp - z_low
        ret[..., -1] = (z_low + z_upp) / 2

        for i in range(len(x)):
            rec = cv2.minAreaRect(np.array(x_2d[i].cpu()))
            
            ret[i, 4:6] = torch.tensor(rec[0])
            ret[i, :2] = torch.tensor(rec[1])
            ret[i, 3] = torch.tensor(np.pi / 180 * rec[2])

        ret = ret.to(x.device)

        scale = ret[:, :3]
        rot = ret[:, 3:4]
        loc = ret[:, 4:]

        return scale, rot, loc

    def step(self, x, y):
        
        scale, rot, loc = self.forward(x)

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.abs(torch.cosine_similarity(radian2vec(rot), radian2vec(y[:, 3: 4]))).mean()
        L3 = self.criterion(loc, y[:, 4:])

        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

class ManualFeature_2d(nn.Module):
    def __init__(self, args):
        
        super(ManualFeature_2d, self).__init__()

        pcd_range = np.array(args.point_cloud_range)

        self.grid_size_2d = (pcd_range[3:5] - pcd_range[:2]) // args.voxel_size_2d + 1
        self.voxel_size_2d = torch.FloatTensor(args.voxel_size_2d)

        low_bound_2d = torch.FloatTensor(pcd_range[:2])
        self.offset_2d = -low_bound_2d + self.voxel_size_2d / 2
        self.size_2d = self.grid_size_2d[0] * self.grid_size_2d[1]

        self.sol = None

        # angs = torch.FloatTensor([(np.pi / args.ang_bins * ang - np.pi / 2) for ang in range(args.ang_bins)])
        angs = torch.FloatTensor([(2 * np.pi / args.ang_bins * ang) for ang in range(args.ang_bins)])
        mats = []

        for ang in angs:
            mats.append([
                [torch.cos(ang), -torch.sin(ang), 0],
                [torch.sin(ang), torch.cos(ang), 0],
                [0, 0, 1]
            ])

        self.mats = torch.FloatTensor(mats)
    
    def load(self, path):
        self.sol = StepNet()
        self.sol.load_state_dict(torch.load(path))

    def extract(self, pcd):
        N = pcd.size(1)

        self.mats = self.mats.to(pcd.device)
        self.offset_2d = self.offset_2d.to(pcd.device)
        self.voxel_size_2d = self.voxel_size_2d.to(pcd.device)

        feature_rot = torch.zeros(self.mats.size(0), pcd.size(0), self.size_2d).to(pcd.device)

        for i in range(len(self.mats)):
            pts = (pcd @ self.mats[i].T)[..., :2] + self.offset_2d
            pts = (pts[..., :2] / self.voxel_size_2d).floor()

            pts = pts[..., 1] + pts[..., 0] * self.grid_size_2d[1]

            pts = pts.long()
            for j in range(pcd.size(0)):
                feature_rot[i, j] = torch.histc(pts[j], bins=self.size_2d, min=-0.5, max=self.size_2d - 0.5)

        feature = feature_rot.permute(1, 2, 0) / pcd.size(1)
        
        return feature
    
    def forward(self, x):
        return x

    def step(self, x, y):
        x = x.permute(0, 2, 1)
        x = self.extract(x)
        x = x.reshape(x.size(0), 21, 21, -1).permute(0, 3, 1, 2)

        scale, rot, loc = self.forward(x)

        L1 = self.criterion(scale, y[:, :3])
        L2 = -torch.cosine_similarity(radian2vec(rot), radian2vec(y[:, 3: 4])).mean()
        L3 = self.criterion(loc, y[:, 4:])

        ret = torch.concat((scale, rot, loc), dim=1)

        return (L1, L2, L3), ret

class ManualFeature_2d_Net(ManualFeature_2d):
    
    def load(self, path):
        self.sol = SplitNet()
        self.sol.load_state_dict(torch.load(path))

    def step(self, x, y):
        x = x.permute(0, 2, 1)
        z1, z2 = x[..., 2].min(axis=1)[0], x[..., 2].max(axis=1)[0]
        x = self.extract(x)
        x = x.reshape(x.size(0), 21, 21, -1).permute(0, 3, 1, 2)

        if z1 != z2:
            return self.sol.step(None, x, y, z2 - z1, (z1 + z2) / 2)

        return self.sol.step(None, x, y)

# test
# import open3d as o3d

# pts = torch.rand(100, 3)

# angs = torch.FloatTensor([(np.pi / 18 * ang - np.pi / 2) for ang in range(18)])

# mats = []

# for ang in angs:
#     mats.append([
#         [torch.cos(ang), -torch.sin(ang), 0],
#         [torch.sin(ang), torch.cos(ang), 0],
#         [0, 0, 1]
#     ])

# mats = torch.FloatTensor(mats)

# for i in range(100):
#     pts[i] = torch.ones(3) * i / 100.0

# for mat in mats:
#     new_pts = pts @ mat.T
#     new_pts = np.array(new_pts)
#     print (type(new_pts), new_pts.shape)
#     new_pts = o3d.utility.Vector3dVector(new_pts)
#     pcd = o3d.geometry.PointCloud(new_pts)
#     FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0,0,0])
#     o3d.visualization.draw_geometries([pcd, FOR])