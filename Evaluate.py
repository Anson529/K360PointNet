import torch

from Datasets import SampleData
from Models import PointNet, PointPillar

import argparse
import numpy as np

import os

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

# from train import getparser

if __name__ == '__main__':
    pass
    # parser = getparser()
    # args = parser.parse_args()

    # model = PointNet(args)