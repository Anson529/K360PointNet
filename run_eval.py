import os

os.system('python Evaluate.py \
    --batch_size 2 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 10 \
    --max_num_points 5000 \
    --where_pretrained experiments/fixscale_2\checkpoint_9.pth \
    --out_dim 7 \
    --type 1 \
    --work_dir experiments/fixscale_2')