import os

os.system('python train_LS.py \
    --batch_size 8 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 20 \
    --max_num_points 5000 \
    --out_dim 7 \
    --num_epochs 40 \
    --type 1 \
    --work_dir experiments_LS/fixscale_size5')