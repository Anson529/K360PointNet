import torch

from Datasets import SampleData, Decompose
from Models import ManualFeature, ManualFeature_rot
from Evaluate import evaluation

import argparse
import numpy as np

import os
import json
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

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

    # writer = SummaryWriter(f'{args.work_dir}/tensorboard')

    if args.type == 0:
        dataset = SampleData(args)
    elif args.type == 1:
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

    # model = PointNet(args).to(args.device)
    # model = ManualFeature(args)
    model = ManualFeature_rot(args)
    # model.load(f'{args.work_dir}/sol.pth')
    # if args.pretrain:

    steps = 0

    for epoch in range(0, args.num_epochs):

        losses = []

        moving_loss = 0
        Feature = None
        out = None

        part = 0

        for idx, data in enumerate(val_loader):
            
            input = data['pts'].to(args.device)
            output = data['output']

            # input = torch.zeros_like(input).to(args.device)
            # print (input.dtype)
            # ret = model(input)
            feature = model.extract(input).to('cpu')
            if Feature is None:
                Feature = feature
                out = output
            else:
                Feature = torch.concat((Feature, feature), dim=0)
                out = torch.concat((out, output), dim=0)
            
            steps += 1

            if steps % 10 == 0 or idx == len(train_loader) - 1:
                with open(f'{args.work_dir}/state.txt', 'w') as f:
                    print('working', epoch, Feature.shape, file=f)
                # print (steps)
                

                torch.save(Feature, f'{args.work_dir}/feature_{epoch}.pth')
                torch.save(out, f'{args.work_dir}/out_{epoch}.pth')
                
                # if (len(out) // 10000) != part:
                #     part = len(out) // 10000
                #     Feature = None
                #     out = None

                # print ('out')
                # break

        with open(f'{args.work_dir}/state.txt', 'w') as f:
            print('success', epoch, Feature.shape, file=f)
        continue

        val_loss, L1, L2, L3 = evaluation(val_loader, model, args)
        logs.append({'epoch': epoch, 'val_loss': val_loss})
        save_log(logs, args.work_dir)

        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalar('val_scale_loss', L1, global_step=epoch)
        writer.add_scalar('val_rot_similarity', L2, global_step=epoch)
        writer.add_scalar('val_loc_loss', L3, global_step=epoch)

        val_Losses.append(val_loss)
        plt.plot(val_Losses)
        plt.savefig(f'{args.work_dir}/val_curve.png')
        plt.cla()
        print ('233', epoch)
    
    # result = {'train': Losses, 'val': val_Losses}

    # with open(f'{args.work_dir}/log.json')
