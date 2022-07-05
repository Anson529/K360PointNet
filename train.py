import torch

from Datasets import SampleData
from Models import PointNet, PointPillar
from Evaluate import evaluation

import argparse
import numpy as np

import os
import json
import matplotlib.pyplot as plt

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

def save_log(logs, work_dir):
    with open(f'{work_dir}/logs.json', 'w') as f:
        f.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in logs) +
            ']\n')   

if __name__ == '__main__':

    parser = getparser()
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    torch.manual_seed(42)

    dataset = SampleData(args)

    train_size = int(len(dataset) * 0.8)

    train_set, val_set = torch.utils.data.dataset.random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        #num_workers=8,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        #num_workers=8,
    )

    model = PointNet(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    Losses = []
    val_Losses = []
    steps = 0

    logs = []

    for epoch in range(0, args.num_epochs):

        losses = []
        model.train()

        for idx, data in enumerate(train_loader):
            
            input = data[0].to(args.device).permute(0, 2, 1)
            output = data[2].to(args.device)
            # print (input.dtype)
            ret = model(input)
            # print (output)
            
            loss = criterion(ret, output)
            loss.backward()

            steps += 1

            if steps % args.grad_cumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            losses.append(loss.item())
            Losses.append(np.mean(losses))
            # print (Losses[-1])

            if idx % 10 == 0:
                
                plt.plot(Losses)
                plt.savefig(f'{args.work_dir}/train_curve.png')
                plt.cla()

                logs.append({'epoch': epoch, 'step': idx, 'loss_mean': Losses[-1]})
                save_log(logs, args.work_dir)

        val_Losses.append(evaluation(val_loader, model, args))
        logs.append({'epoch': epoch, 'val_loss': val_Losses[-1]})
        save_log(logs, args.work_dir)

        plt.plot(val_Losses)
        plt.savefig(f'{args.work_dir}/val_curve.png')
        plt.cla()
    
    # result = {'train': Losses, 'val': val_Losses}

    # with open(f'{args.work_dir}/log.json')
