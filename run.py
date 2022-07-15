import os

os.system('python train.py \
    --batch_size 2 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 10 \
    --max_num_points 5000 \
    --out_dim 7 \
    --work_dir experiments/transformation')