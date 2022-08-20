from msilib.schema import Feature
import torch

from Datasets import SampleData, Decompose
from Models import ManualFeature
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
    model = ManualFeature(args)
    model.load(f'{args.work_dir}/sol.pth')
    # if args.pretrain:

    Losses = []
    val_Losses = []
    steps = 0

    logs = []

    Feature = torch.load(f'{args.work_dir}/feature.pth')
    Output = torch.load(f'{args.work_dir}/out.pth')

    Feature = Feature.reshape(Feature.size(0), -1) / 5000

    print (Feature.shape)
    print(Output.shape)

    # # Feature = Feature.to(args.device)
    # # Output = Output.to(args.device)

    torch.manual_seed(102)

    out = torch.linalg.lstsq(Feature[:10000], Output[:10000]).solution

    criterion = torch.nn.MSELoss()
    # torch.save(out, f'{args.work_dir}/sol_all.pth')
    out = torch.load(f'{args.work_dir}/sol_all.pth')
    print (out[0])
    input()
    # quit()
    Loss = []
    LL1, LL2, LL3 = [], [], []
    for i in range(10000, 20000):
        A, B = Feature[i] @ out, Output[i]

        L1 = criterion(A[:3], B[:3])
        L2 = criterion(A[3], B[3])
        L3 = criterion(A[4:], B[4:])

        LL1.append(L1.item())
        LL2.append(L2.item())
        LL3.append(L3.item())
        # quit()
        print (np.mean(LL1), np.mean(LL2), np.mean(LL3))

    quit()

    for epoch in range(0, args.num_epochs):

        losses = []

        moving_loss = 0
        Feature = None
        out = None

        # for idx, data in enumerate(train_loader):
            
        #     input = data['pts'].to(args.device)
        #     output = data['output']

        #     # input = torch.zeros_like(input).to(args.device)
        #     # print (input.dtype)
        #     # ret = model(input)
        #     feature = model.extract(input).to('cpu')
        #     if Feature is None:
        #         Feature = feature
        #         out = output
        #     else:
        #         Feature = torch.concat((Feature, feature), dim=0)
        #         out = torch.concat((out, output), dim=0)
            
        #     steps += 1

        #     if steps % 10 == 0:
        #         print (steps)
        #         print (Feature.shape)
        #         print (out.shape)
        #         torch.save(Feature, f'{args.work_dir}/feature.pth')
        #         torch.save(out, f'{args.work_dir}/out.pth')
        #         print ('out')

        #     continue

        #     if steps % args.grad_cumulate == 0:
        #         writer.add_scalar('scale_grad', torch.norm(model.scaleNet.weight.grad).item(), global_step=steps)
        #         writer.add_scalar('rot_grad', torch.norm(model.rotNet.weight.grad).item(), global_step=steps)
        #         writer.add_scalar('loc_grad', torch.norm(model.locNet.weight.grad).item(), global_step=steps)
        #         optimizer.step()
        #         optimizer.zero_grad()
                
        #     losses.append(loss.item())

        #     moving_loss = moving_loss * 0.98 + loss.item() * 0.02

        #     Losses.append(np.mean(losses))
        #     # print (Losses[-1])
        #     writer.add_scalar('train_loss', loss.item(), global_step=steps)
        #     writer.add_scalar('scale_loss', ret[0].item(), global_step=steps)
        #     writer.add_scalar('rot_similarity', ret[1].item(), global_step=steps)
        #     writer.add_scalar('loc_loss', ret[2].item(), global_step=steps)

        #     if idx % 10 == 0:
                
        #         plt.plot(Losses)
        #         plt.savefig(f'{args.work_dir}/train_curve.png')
        #         plt.cla()

        #         print (steps, moving_loss)
                
        #         logs.append({'epoch': epoch, 'step': idx, 'loss_mean': Losses[-1], 'moving_ave': moving_loss})
        #         save_log(logs, args.work_dir)

        #         torch.save(model.state_dict(), f'{args.work_dir}/checkpoint_{epoch}.pth')

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
