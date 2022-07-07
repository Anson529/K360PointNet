import torch

from Datasets import SampleData
from Models import PointNet, PointPillar
from Geometry import visualize_sample, test_sample

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
        output = data[-1].to(args.device)
        # print (input.dtype)
        ret = model(input)

        loss = criterion(ret, output)
        losses.append(loss.item())

    return np.mean(losses)

def visualize(val_set, val_loader, model, args):
    model.eval()
    print ('visualizing')
    criterion = torch.nn.MSELoss()

    losses = []

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        output = data['output'].to(args.device)
        # print (input.dtype)
        ret = model(input)

        loss = criterion(ret, output)
        losses.append(loss.item())

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            visualize_sample(data['pts'][i], data['R'][i], data['T'][i], ret[i], args)

    return np.mean(losses)

def process(val_set, val_loader, model, args):
    model.eval()
    print ('processing')
    criterion = torch.nn.MSELoss()

    losses = []

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        output = data['output'].to(args.device)
        # print (input.dtype)
        ret = model(input)

        loss = criterion(ret, output)
        losses.append(loss.item())

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            test_sample(data['output'][i], ret[i], data['scales'][i], data['trans'][i], data['pcd_path'][i], args)

    return np.mean(losses)

from Options import getparser

if __name__ == '__main__':
    # pass
    parser = getparser()
    args = parser.parse_args()

    model = PointNet(args).to(args.device)
    model.load_state_dict(torch.load(args.where_pretrained))

    torch.manual_seed(42)

    dataset = SampleData(args)

    train_size = int(len(dataset) * 0.8)

    train_set, val_set = torch.utils.data.dataset.random_split(dataset, [train_size, len(dataset) - train_size])

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        #num_workers=8,
    )

    process(val_set, val_loader, model, args)

    print (model)