import os

os.system('python Evaluate.py \
    --batch_size 2 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 10 \
    --max_num_points 5000 \
    --where_pretrained experiments/transformation\checkpoint_0.pth \
    --out_dim 7 \
    --type 1 \
    --work_dir experiments/transformation')