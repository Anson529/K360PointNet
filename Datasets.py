from matplotlib import scale
from matplotlib.pyplot import box
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle
import os
import numpy as np

class SampleData(Dataset):

    def __init__(self, args):
        
        self.point_cloud_range = args.point_cloud_range
        self.data_path = args.data_path
        self.eps = args.eps
        self.max_num_points = args.max_num_points

        np.random.seed(42)

        with open(args.info_path, 'rb') as f:
            self.data_info = pickle.load(f)

    def __len__(self):
        return len(self.data_info)
    
    def getbox(self, bbox):
        conf = np.random.uniform(0, 1, 6)
        conf[: 3] = - conf[: 3]
        size = bbox[3:] - bbox[:3]
        size = np.concatenate((size, size), axis=0)
        return bbox + conf * self.eps * size

    def __getitem__(self, idx):
        sample_info = self.data_info[idx]

        box_bound = sample_info['bbox']
        radius = (box_bound[3:] - box_bound[:3]) / 2

        # enlarge the box & zero-center
        box_bound = self.getbox(box_bound) #+ [-1, -1, -1, 1, 1, 1]
        box_center = (box_bound[:3] + box_bound[3:]) / 2
        
        # fetch the points form the point cloud
        pcd = np.load(os.path.join(self.data_path, sample_info['pcd_path']))[:, :3]
        
        bound_x = np.logical_and(pcd[:, 0] > box_bound[0], pcd[:, 0] < box_bound[3])
        bound_y = np.logical_and(pcd[:, 1] > box_bound[1], pcd[:, 1] < box_bound[4])
        bound_z = np.logical_and(pcd[:, 2] > box_bound[2], pcd[:, 2] < box_bound[5])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        pcd = pcd[bb_filter] - box_center

        R = sample_info['R']
        T = sample_info['T'] - box_center

        pts = np.zeros((self.max_num_points, 3))

        # get a fix number of points
        np.random.shuffle(pcd)
        if len(pcd):
            for i in range(self.max_num_points):
                pts[i] = pcd[i % len(pcd)]

        # rescale the box to [-10, 10] ^ 3
        box_bound[3: ] -= box_center
        scales = np.array([1, 1, 1])          
        for i in range(3):
            if box_bound[3 + i] > 0:
                scales[i] = self.point_cloud_range[3 + i] / box_bound[3 + i]

        radius = np.max(radius * scales)
        out = np.concatenate((T * scales, [radius]), axis=0)

        pts = torch.FloatTensor(pts * scales)
        R = torch.FloatTensor(R * scales)
        T = torch.FloatTensor(T * scales)
        out = torch.FloatTensor(out)

        # return pts, R, T, out
        return {"pts": pts, "output": out, "R": R, "T": T, "scales": scales, "trans": box_center, "pcd_path": sample_info['pcd_path']}
        
from Options import getparser

if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = SampleData(args)
    
    for i in range(200):
        pts, R, T, out = dataset[i]
        # visualize_sample(pts, R, T, out)
