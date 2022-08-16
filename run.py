import os

os.system('python train.py \
    --batch_size 4 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 20 \
    --max_num_points 5000 \
    --out_dim 7 \
    --num_epochs 40 \
    --type 1 \
    --data_path /projects/perception/personals/wenjiey/works/data/trans \
    --info_path /projects/perception/personals/wenjiey/works/data/trans/info.pkl \
    --work_dir experiments_40/fixscale_0.2noscale')

    # --data_path /projects/perception/personals/wenjiey/works/data/building/data \
    # --info_path /projects/perception/personals/wenjiey/works/data/building/data/info.pkl \

#  --data_path /projects/perception/personals/wenjiey/works/data/trans \
#     --info_path /projects/perception/personals/wenjiey/works/data/trans/info.pkl \