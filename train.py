import torch

from Datasets import SampleData
from Models import PointNet, PointPillar

import argparse
import numpy as np

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--info_path', type=str, default='data/info.pkl')

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

    return parser

if __name__ == '__main__':

    parser = getparser()
    args = parser.parse_args()

    torch.manual_seed(42)

    dataset = SampleData(args.data_path, args.info_path, args.max_num_points, args.eps)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        #num_workers=8,
    )

    model = PointNet(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    Losses = []
    steps = 0

    for epoch in range(0, args.num_epochs):

        losses = []

        for idx, data in enumerate(loader):
            
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
            print (Losses[-1])

            if idx % 10 == 0:
                import matplotlib.pyplot as plt
                plt.plot(Losses)
                plt.savefig('curve.png')
                plt.cla()