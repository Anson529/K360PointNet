import os

os.system('python train_LS.py \
    --batch_size 16 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 20 \
    --max_num_points 5000 \
    --out_dim 7 \
    --num_epochs 1 \
    --type 1 \
    --voxel_size 2 2 2 \
    --work_dir experiments_other/garage/fixscale_size2_rotfeat_val \
    --data_path /projects/perception/personals/wenjiey/works/data/garage/data \
    --info_path /projects/perception/personals/wenjiey/works/data/garage/data/info.pkl \
    ')
       
# --data_path /projects/perception/personals/wenjiey/works/data/trans \
#     --info_path /projects/perception/personals/wenjiey/works/data/trans/info.pkl \
    # --data_path /projects/perception/personals/wenjiey/works/data/building/data \
    # --info_path /projects/perception/personals/wenjiey/works/data/building/data/info.pkl \