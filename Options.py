import argparse

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='E:\work\kitti360\code\processed/vegetation/trans')
    parser.add_argument('--info_path', type=str, default='E:\work\kitti360\code\processed/vegetation/trans\info.pkl')

    parser.add_argument('--voxel_size', type=list, default=[1, 1, 1])
    parser.add_argument('--point_cloud_range', type=list, default=[-10, -10, -10, 10, 10, 10])
    parser.add_argument('--max_num_points_voxel', type=int, default=100)
    parser.add_argument('--max_num_points', type=int, default=5000)
    parser.add_argument('--eps', type=float, default=0)

    parser.add_argument('--type', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--grad_cumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w', type=list, default=[1, 10, 1])

    parser.add_argument('--out_dim', type=int, default=4)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--work_dir', type=str, default='experiments/test')

    parser.add_argument('--pretrain', type=bool, default='False')
    parser.add_argument('--where_pretrained', type=str, default='experiments/test/checkpoint.pth')

    return parser