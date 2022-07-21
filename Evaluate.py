import torch

from Datasets import Decompose, SampleData
from Models import PointNet, PointPillar
from Geometry import test_sample_dec, visualize_sample, test_sample

import argparse
import numpy as np

import os
import pickle

def evaluation(val_loader, model, args):
    # model.eval()
    print ('evaluating')
    criterion = torch.nn.MSELoss()

    losses = []

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        output = data['output'].to(args.device)

        # input = torch.zeros_like(input).to(args.device)
        # print (input.dtype)
        ret = model(input)
        # print (ret[0])
        # quit()

        loss = criterion(ret, output)
        losses.append(loss.item())

        print (np.mean(losses))

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
    import open3d as o3d

    # model.eval()
    print ('processing')
    criterion = torch.nn.MSELoss()

    losses = []

    results = []
    cnt = {}

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        
        output = data['output'].to(args.device)
        # input = torch.zeros_like(input).to(args.device)
        # print (input.dtype)
        ret = model(input)

        loss = criterion(ret, output)
        losses.append(loss.item())
        # print (ret[0])
        # print (ret[1])
        # quit()

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            sample_path = data['pcd_path'][i][:-8]
            mesh = test_sample_dec(data['output'][i], ret[i], data['scales'][i], data['trans'][i], args)

            if sample_path in cnt:
                cnt[sample_path] += 1
            else:
                cnt[sample_path] = 0

            os.makedirs(f'{args.work_dir}/result/{sample_path}/gt', exist_ok=True)
            os.makedirs(f'{args.work_dir}/result/{sample_path}/pre', exist_ok=True)

            o3d.io.write_triangle_mesh(f'{args.work_dir}/result/{sample_path}/gt/{cnt[sample_path]}.ply', mesh[0])
            o3d.io.write_triangle_mesh(f'{args.work_dir}/result/{sample_path}/pre/{cnt[sample_path]}.ply', mesh[1])
            # results.append(ret)

    # with open(f'{args.work_dir}/result.pkl,)

    return np.mean(losses)

from Options import getparser

if __name__ == '__main__':
    # pass
    parser = getparser()
    args = parser.parse_args()

    model = PointNet(args).to(args.device)
    model.load_state_dict(torch.load(args.where_pretrained))

    torch.manual_seed(42)

    if args.type == 0:
        dataset = SampleData(args)
    elif args.type == 1:
        dataset = Decompose(args)

    train_size = int(len(dataset) * 0.8)

    train_set, val_set = torch.utils.data.dataset.random_split(dataset, [train_size, len(dataset) - train_size])

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        #num_workers=8,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        #num_workers=8,
    )
    # visualize(dataset, loader, model, args)
    # evaluation(val_loader, model, args)
    process(dataset, loader, model, args)

    print (model)