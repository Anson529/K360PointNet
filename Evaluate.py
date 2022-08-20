import torch

from Datasets import Decompose, SampleData
from Models import PointNet, PointPillar, PointNetV2, ManualFeature
from Geometry import test_sample_dec, vis_sample, visualize_sample, test_sample

import argparse
import numpy as np

import os
import pickle

from tqdm import tqdm

def evaluation(val_loader, model, args):
    # model.eval()
    print ('evaluating')
    criterion = torch.nn.MSELoss()

    losses = []
    L1, L2, L3 = [], [], []

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        output = data['output'].to(args.device)

        # input = torch.zeros_like(input).to(args.device)
        # print (input.dtype)
        # ret = model(input)

        ret = model.step(input, output)[0]

        loss = ret[0] * args.w[0] - ret[1] * args.w[1] + ret[2] * args.w[2]
        losses.append(loss.item())
        L1.append(ret[0].item())
        L2.append(ret[1].item())
        L3.append(ret[2].item())

        print (np.mean(L1), np.mean(L2), np.mean(L3))

    return np.mean(losses), np.mean(L1), np.mean(L2), np.mean(L3)

def visualize(val_set, val_loader, model, args):
    # model.eval()
    print ('visualizing')
    criterion = torch.nn.MSELoss()

    losses = []
    cnt = {}

    import open3d as o3d

    for idx, data in enumerate(val_loader):
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        output = data['output'].to(args.device)
        loss, ret = model.step(input, output)

        loss = criterion(ret, output)
        losses.append(loss.item())

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            sample_path = data['pcd_path'][i][:-8]
            print (sample_path)
            pcd, gt_mesh, mesh, bbox_line = vis_sample(data['pts'][i], data['output'][i], ret[i], args)
            # visualize_sample(data['pts'][i], data['R'][i], data['T'][i], ret[i], args)

            if sample_path in cnt:
                cnt[sample_path] += 1
            else:
                cnt[sample_path] = 0

            os.makedirs(f'{args.work_dir}/origin/{sample_path}/{cnt[sample_path]}', exist_ok=True)
            
            o3d.io.write_point_cloud(f'{args.work_dir}/origin/{sample_path}/{cnt[sample_path]}/pcd.ply', pcd)
            o3d.io.write_triangle_mesh(f'{args.work_dir}/origin/{sample_path}/{cnt[sample_path]}/gt.ply', gt_mesh)
            o3d.io.write_triangle_mesh(f'{args.work_dir}/origin/{sample_path}/{cnt[sample_path]}/pre.ply', mesh)
            o3d.io.write_line_set(f'{args.work_dir}/origin/{sample_path}/{cnt[sample_path]}/box.ply', bbox_line)
        
        

    return np.mean(losses)

def calcIoU(val_set, val_loader, model, args):
    import open3d as o3d

    # model.eval()
    print ('calculating 3d IoU')
    criterion = torch.nn.MSELoss()

    losses = []

    results = []
    cnt = {}

    IoUs = []

    print ('total:', len(val_loader))

    for idx, data in enumerate(val_loader):
        if idx % 10 == 0:
            print (idx)
            
        input = data['pts'].to(args.device).permute(0, 2, 1)
        # input = torch.zeros_like(input).to(args.device)
        output = data['output'].to(args.device)
        loss, ret = model.step(input, output)

        loss = loss[0] * args.w[0] - loss[1] * args.w[1] + loss[2] * args.w[2]

        losses.append(loss.item())

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            # ret[i, 3] = torch.arcsin(ret[i, 3])
            sample_path = data['pcd_path'][i][:-8]
            mesh = test_sample_dec(data['output'][i], ret[i], data['scales'][i], data['trans'][i], args)

            o3d.io.write_triangle_mesh("clibs/std.ply", mesh[0])
            o3d.io.write_triangle_mesh("clibs/predicted.ply", mesh[1])
            
            try:
                os.system("cd clibs && 3dIoU")
                with open("clibs/IoU.out", "r") as f:
                    IoU = float(f.readline())
            
                IoUs.append(IoU)
            except:
                pass
            # results.append(ret)
        print (np.mean(IoUs))

        if idx == 1000:
            break
    # with open(f'{args.work_dir}/result.pkl,)

    return np.mean(IoUs)

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
        # ret = model(input)

        # loss = criterion(ret, output)
        # 

        loss, ret = model.step(input, output)

        loss = loss[0] * args.w[0] - loss[1] * args.w[1] + loss[2] * args.w[2]

        losses.append(loss.item())

        ret = ret.detach().to('cpu')
        for i in range(args.batch_size):
            sample_path = data['pcd_path'][i][:-8]
            # ret[i, 3] = torch.arcsin(ret[i, 3])
            print (ret[i, 3], sample_path)
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

    # model = PointNetV2(args).to(args.device)
    # model.load_state_dict(torch.load(args.where_pretrained))
    model = ManualFeature(args).to(args.device)
    model.load(args.where_pretrained)

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
    visualize(dataset, loader, model, args)
    # evaluation(val_loader, model, args)
    # process(dataset, loader, model, args)
    # IoU = calcIoU(train_loader, val_loader, model, args)

    # with open(f'{args.work_dir}/IoU.txt', 'w') as f:
    #     print (IoU, file=f)

    print (model)