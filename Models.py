import torch
import torch.nn as nn
import numba
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torchvision.models import resnet18
from pointnet.model import PointNetfeat


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
        self.sol = torch.load(path)

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

class VoxelGenerator(object):
    """Voxel generator in numpy implementation.

    Args:
        voxel_size (list[float]): Size of a single voxel
        point_cloud_range (list[float]): Range of points
        max_num_points (int): Maximum number of points in a single voxel
        max_voxels (int, optional): Maximum number of voxels.
            Defaults to 20000.
    """

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points):
        """Generate voxels given points."""
        return points_to_voxel(points, self._voxel_size,
                               self._point_cloud_range, self._max_num_points,
                               False, self._max_voxels)

    @property
    def voxel_size(self):
        """list[float]: Size of a single voxel."""
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        """int: Maximum number of points per voxel."""
        return self._max_num_points

    @property
    def point_cloud_range(self):
        """list[float]: Range of point cloud."""
        return self._point_cloud_range

    @property
    def grid_size(self):
        """np.ndarray: The size of grids."""
        return self._grid_size

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        indent = ' ' * (len(repr_str) + 1)
        repr_str += f'(voxel_size={self._voxel_size},\n'
        repr_str += indent + 'point_cloud_range='
        repr_str += f'{self._point_cloud_range.tolist()},\n'
        repr_str += indent + f'max_num_points={self._max_num_points},\n'
        repr_str += indent + f'max_voxels={self._max_voxels},\n'
        repr_str += indent + f'grid_size={self._grid_size.tolist()}'
        repr_str += ')'
        return repr_str


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        max_points (int): Indicate maximum points contained in a voxel.
        reverse_index (bool): Whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels (int): Maximum number of voxels this function creates.
            For second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                            num_points_per_voxel,
                                            coor_to_voxelidx, voxels, coors,
                                            max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    num_points_per_voxel,
                                    coor_to_voxelidx,
                                    voxels,
                                    coors,
                                    max_points=35,
                                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num

