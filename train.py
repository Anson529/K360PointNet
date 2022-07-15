import torch

from Datasets import SampleData, Decompose
from Models import PointNet, PointPillar
from Evaluate import evaluation

import argparse
import numpy as np

import os
import json
import matplotlib.pyplot as plt

from Options import getparser

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

    # dataset = SampleData(args)
    dataset = Decompose(args)

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
        center_losses = []
        center = torch.zeros(args.batch_size, 4).to(args.device)
        center[:, 3] = 6
        model.train()

        moving_loss = 0

        for idx, data in enumerate(train_loader):
            
            input = data['pts'].to(args.device).permute(0, 2, 1)
            output = data['output'].to(args.device)
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

            moving_loss = moving_loss * 0.95 + loss.item() * 0.05

            Losses.append(np.mean(losses))
            # print (Losses[-1])

            if idx % 10 == 0:
                
                plt.plot(Losses)
                plt.savefig(f'{args.work_dir}/train_curve.png')
                plt.cla()

                # if len(Losses) > 1 and Losses[-1] - Losses[-2] > 1:
                #     print ('error message', output)

                print (Losses[-1], moving_loss)
                
                logs.append({'epoch': epoch, 'step': idx, 'loss_mean': Losses[-1]})
                save_log(logs, args.work_dir)

                torch.save(model.state_dict(), f'{args.work_dir}/checkpoint_{epoch}.pth')

        val_Losses.append(evaluation(val_loader, model, args))
        logs.append({'epoch': epoch, 'val_loss': val_Losses[-1]})
        save_log(logs, args.work_dir)

        plt.plot(val_Losses)
        plt.savefig(f'{args.work_dir}/val_curve.png')
        plt.cla()
    
    # result = {'train': Losses, 'val': val_Losses}

    # with open(f'{args.work_dir}/log.json')
