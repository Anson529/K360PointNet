import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle
import os
import numpy as np

class SampleData(Dataset):

    def __init__(self, data_path, info_path, max_points=20000, eps=0):
        
        self.data_path = data_path
        self.eps = eps
        self.max_points = max_points

        with open(info_path, 'rb') as f:
            self.data_info = pickle.load(f)

    def __len__(self):
        return len(self.data_info)
    
    def getbox(self, bbox):
        conf = np.random.uniform(0, 1, 6)
        conf[: 3] = - conf[: 3]
        size = bbox[3:] - bbox[:3]
        size = np.concatenate((size, size), axis=0)
        return bbox + conf * self.eps * size

    def visualize(self, idx):
        pts, R, T = self.__getitem__(idx)

        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(os.path.join(self.data_path, 'std.ply'))
        points = np.array(mesh.vertices) @ R + T
        mesh.vertices = o3d.utility.Vector3dVector(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        FOR = o3d.geometry.TriangleMesh.create_coordinate_frame()

        o3d.visualization.draw_geometries([pcd, FOR])
        o3d.visualization.draw_geometries([mesh, pcd, FOR])

    def __getitem__(self, idx):
        sample_info = self.data_info[idx]

        box_bound = self.getbox(sample_info['bbox']) #+ [-1, -1, -1, 1, 1, 1]
        box_center = (box_bound[:3] + box_bound[3:]) / 2

        pcd = np.load(os.path.join(self.data_path, sample_info['pcd_path']))[:, :3]
        
        bound_x = np.logical_and(pcd[:, 0] > box_bound[0], pcd[:, 0] < box_bound[3])
        bound_y = np.logical_and(pcd[:, 1] > box_bound[1], pcd[:, 1] < box_bound[4])
        bound_z = np.logical_and(pcd[:, 2] > box_bound[2], pcd[:, 2] < box_bound[5])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        pcd = pcd[bb_filter] - box_center

        R = sample_info['R']
        T = sample_info['T'] - box_center

        pts = np.zeros((self.max_points, 3))
        np.random.shuffle(pcd)

        for i in range(self.max_points):
            if i < len(pcd):
                pts[i] = pcd[i]
            # else:
            #     pts[i] = [100, 100, 100]
        
        # print (idx, len(pcd))
        pts = torch.FloatTensor(pts)

        return pts, R, T
        
        

if __name__ == '__main__':
    data_path = 'E:\work\kitti360\code\processed/vegetation\grid'
    info_path = 'E:\work\kitti360\code\processed/vegetation\data/info.pkl'
    
    # print (233)
    dataset = SampleData(data_path, info_path, eps=0)
    
    for i in range(200):
        dataset.visualize(5715)