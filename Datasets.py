import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pickle
import os
import numpy as np
from scipy.spatial.transform import Rotation as sc_R

from Geometry import *

class SampleData(Dataset):

    def __init__(self, args):
        
        self.point_cloud_range = args.point_cloud_range
        self.data_path = args.data_path
        self.eps = args.eps
        self.random_rot = args.random_rot
        self.max_num_points = args.max_num_points
        self.ambig = args.ambig

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
        scales = np.array([1.0, 1.0, 1.0])          
        for i in range(3):
            if box_bound[3 + i] > 0:
                scales[i] = self.point_cloud_range[3 + i] / box_bound[3 + i]

        radius = np.max(radius * scales)
        out = np.concatenate((T * scales, [radius]), axis=0)

        pts = torch.FloatTensor(pts * scales)
        R = torch.FloatTensor(R * scales)
        T = torch.FloatTensor(T * scales)
        out = torch.FloatTensor(out)

        print (scales)

        # return pts, R, T, out
        return {"pts": pts, "output": out, "R": R, "T": T, "scales": scales, "trans": box_center, "pcd_path": sample_info['pcd_path']}

class Decompose(SampleData):
    
    def getbox(self, bbox):
        conf = np.random.uniform(0, 1, 6)
        conf[: 3] = - conf[: 3]
        size = bbox[3:] - bbox[:3]
        size = np.concatenate((size, size), axis=0)
        return bbox + conf * self.eps * size

    def __getitem__(self, idx):
        sample_info = self.data_info[idx]
        box_bound = sample_info['bbox']

        # enlarge the box & zero-center
        box_bound = self.getbox(box_bound) #+ [-1, -1, -1, 1, 1, 1]
        box_center = (box_bound[:3] + box_bound[3:]) / 2
        
        # fetch the points form the point cloud
        pcd = np.load(os.path.join(self.data_path, sample_info['pcd_path']))

        # filter samples with too few points
        # if len(pcd) < 100000:
        #     return self.__getitem__(np.random.randint(self.__len__()))
        seq = []
        for i in range(self.__len__()):
            sample_info = self.data_info[i]
            pcd = np.load(os.path.join(self.data_path, sample_info['pcd_path']))
            bbox = sample_info['bbox']
            box_size = (bbox[3:] - bbox[:3]).max()
            
            # seq.append(len(pcd))
            seq.append(box_size)
        print (self.__len__())
        import matplotlib.pyplot as plt
        # feaq = plt.hist(seq, bins=[0, 10, 100, 1000, 10000, 100000, 1000000])
        feaq = plt.hist(seq)
        print (feaq)
        ax = plt.gca()
        # ax.set_xscale('log')
        # plt.savefig('pts.png')
        plt.show()
        quit()
        
        bound_x = np.logical_and(pcd[:, 0] >= box_bound[0], pcd[:, 0] <= box_bound[3])
        bound_y = np.logical_and(pcd[:, 1] >= box_bound[1], pcd[:, 1] <= box_bound[4])
        bound_z = np.logical_and(pcd[:, 2] >= box_bound[2], pcd[:, 2] <= box_bound[5])

        if pcd.shape[1] == 5:
            bound_x = np.logical_and(bound_x, pcd[:, 3] == sample_info['glo_ID'])

        pcd = pcd[:, :3]

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

        pcd = pcd[bb_filter] - box_center

        R = sample_info['R']
        T = sample_info['T'] - box_center

        # rescale the box to [-10, 10] ^ 3
        box_bound[3: ] -= box_center
        scales = np.array([1.0, 1.0, 1.0])          
        for i in range(3):
            if box_bound[3 + i] > 0:
                scales[i] = self.point_cloud_range[3 + i] / box_bound[3 + i]

        SCALE = scales.min()
        scales = np.array([SCALE, SCALE, SCALE])

        if self.random_rot:
            rot = randrot()
        else:
            rot = np.identity(3)

        aug = scales * rot

        out = np.concatenate((decomposition(aug @ R), T @ aug.T), axis=0)

        # for undirected meshes
        if self.ambig == False:
            if out[0] < out[1]:
                out[0], out[1] = out[1], out[0]
                out[3] += np.pi / 2
            
            if out[3] < 0:
                out[3] += 2 * np.pi

            while out[3] > 0.5 * np.pi:
                out[3] -= np.pi

        pcd = pcd @ aug.T
        bound_x = np.logical_and(pcd[:, 0] > self.point_cloud_range[0], pcd[:, 0] < self.point_cloud_range[3])
        bound_y = np.logical_and(pcd[:, 1] > self.point_cloud_range[1], pcd[:, 1] < self.point_cloud_range[4])
        bound_z = np.logical_and(pcd[:, 2] > self.point_cloud_range[2], pcd[:, 2] < self.point_cloud_range[5])

        bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        pcd = pcd[bb_filter]

        # get a fix number of points
        if self.max_num_points:
            pts = np.zeros((self.max_num_points, 3))
            np.random.shuffle(pcd)
            if len(pcd):
                for i in range(self.max_num_points):
                    pts[i] = pcd[i % len(pcd)]
        else:
            pts = pcd

        pts = torch.FloatTensor(pts)
        R = torch.FloatTensor(aug @ R)
        T = torch.FloatTensor(T @ aug.T)
        out = torch.FloatTensor(out)

        return {"pts": pts, "output": out, "R": R, "T": T, "scales": scales, "trans": box_center, "pcd_path": sample_info['pcd_path'], "aug": aug}
        
from Options import getparser

if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = Decompose(args)
    
    print (len(dataset))