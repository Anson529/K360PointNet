import os

os.system('python Datasets.py \
    --batch_size 4 \
    --device cuda \
    --eps 0 \
    --grad_cumulate 10 \
    --max_num_points 5000')