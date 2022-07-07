import os

os.system('python Evaluate.py \
    --batch_size 1 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 10 \
    --max_num_points 5000 \
    --where_pretrained experiments\sphere\checkpoint_7.pth')