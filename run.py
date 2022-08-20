import os

os.system('python train.py \
    --batch_size 2 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 20 \
    --max_num_points 5000 \
    --out_dim 7 \
    --num_epochs 40 \
    --type 1 \
    --work_dir experiments_40/fixscale_1')

# --data_path E:\work\kitti360\code\processed/building/data \
#     --info_path E:\work\kitti360\code\processed/building/data/info.pkl \