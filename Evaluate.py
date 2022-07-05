import torch

from Datasets import SampleData
from Models import PointNet, PointPillar

import argparse
import numpy as np

import os

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='E:\work\kitti360\code\processed/vegetation/trans')
    parser.add_argument('--info_path', type=str, default='E:\work\kitti360\code\processed/vegetation/trans\info.pkl')

    parser.add_argument('--voxel_size', type=list, default=[0.1, 0.1, 20])
    parser.add_argument('--point_cloud_range', type=list, default=[-10, -10, -10, 10, 10, 10])
    parser.add_argument('--max_num_points_voxel', type=int, default=100)
    parser.add_argument('--max_num_points', type=int, default=5000)
    parser.add_argument('--eps', type=float, default=0)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--grad_cumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--work_dir', type=str, default='experiments/test')

    return parser

def evaluation(val_loader, model, args):
    model.eval()
    print ('evaluating')
    criterion = torch.nn.MSELoss()

    losses = []

    for idx, data in enumerate(val_loader):
            
        input = data[0].to(args.device).permute(0, 2, 1)
        output = data[2].to(args.device)
        # print (input.dtype)
        ret = model(input)

        loss = criterion(ret, output)
        losses.append(loss.item())

    return np.mean(losses)

if __name__ == '__main__':
    pass