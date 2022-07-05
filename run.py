import os

os.system('python train.py \
    --data_path E:\work\kitti360\code\processed/vegetation\grid \
    --info_path E:\work\kitti360\code\processed/vegetation\data\info.pkl \
    --batch_size 2 \
    --device cuda \
    --eps 0.3 \
    --grad_cumulate 10 \
    --max_num_points 5000')