import os

os.system('python train.py \
    --batch_size 128 \
    --device cuda \
    --eps 0.6 \
    --grad_cumulate 1 \
    --max_num_points 5000 \
    --out_dim 7 \
    --num_epochs 200 \
    --type 1 \
    --w 1 1 1 \
    --voxel_size 1 1 1 \
    --lr 1e-5 \
    --where_pretrained experiments_LS/fixscale_size1_norot_conv_100e_lrdecay_vec_drop0.5/checkpoint_best.pth \
    --data_path /projects/perception/personals/wenjiey/works/data/trans \
    --info_path /projects/perception/personals/wenjiey/works/data/trans/info.pkl \
    --work_dir experiments_LS/fixscale_size1_norot_conv_200e_lrdecay_pretrain')

    # --data_path /projects/perception/personals/wenjiey/works/data/building/data \
    # --info_path /projects/perception/personals/wenjiey/works/data/building/data/info.pkl \

#  --data_path /projects/perception/personals/wenjiey/works/data/trans \
#     --info_path /projects/perception/personals/wenjiey/works/data/trans/info.pkl \