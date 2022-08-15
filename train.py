import torch

from Datasets import SampleData, Decompose
from Models import PointNet, PointPillar, PointNetV2
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

    writer = SummaryWriter(f'{args.work_dir}/tensorboard')

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
    model = PointNetV2(args).to(args.device)
    # if args.pretrain:


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
        moving_loss_1, moving_loss_2, moving_loss_3 = 0, 0, 0

        ori_Min, ori_Max = 10, -10

        for idx, data in enumerate(train_loader):
            
            input = data['pts'].to(args.device).permute(0, 2, 1)
            output = data['output'].to(args.device)

            # input = torch.zeros_like(input).to(args.device)
            # print (input.dtype)
            # ret = model(input)


            ret = model.step(input, output)[0]

            loss = ret[0] * args.w[0] - ret[1] * args.w[1] + ret[2] * args.w[2]

            loss.backward()

            steps += 1

            if steps % args.grad_cumulate == 0:
                writer.add_scalar('scale_grad', torch.norm(model.scaleNet.weight.grad).item(), global_step=steps)
                writer.add_scalar('rot_grad', torch.norm(model.rotNet.weight.grad).item(), global_step=steps)
                writer.add_scalar('loc_grad', torch.norm(model.locNet.weight.grad).item(), global_step=steps)
                optimizer.step()
                optimizer.zero_grad()
                
            losses.append(loss.item())

            moving_loss = moving_loss * 0.98 + loss.item() * 0.02

            Losses.append(np.mean(losses))
            # print (Losses[-1])
            writer.add_scalar('train_loss', loss.item(), global_step=steps)
            writer.add_scalar('scale_loss', ret[0].item(), global_step=steps)
            writer.add_scalar('rot_similarity', ret[1].item(), global_step=steps)
            writer.add_scalar('loc_loss', ret[2].item(), global_step=steps)

            if idx % 10 == 0:
                
                plt.plot(Losses)
                plt.savefig(f'{args.work_dir}/train_curve.png')
                plt.cla()

                print (steps, moving_loss)
                
                logs.append({'epoch': epoch, 'step': idx, 'loss_mean': Losses[-1], 'moving_ave': moving_loss})
                save_log(logs, args.work_dir)

                torch.save(model.state_dict(), f'{args.work_dir}/checkpoint_{epoch}.pth')

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
