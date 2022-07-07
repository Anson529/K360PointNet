import numpy as np
import os
import open3d as o3d
from regex import F

from Geometry import npy2pcd

def get_meshes(path):
    files = os.listdir(path)

    meshes = []
    for file in files:
        meshes.append(o3d.io.read_triangle_mesh(f'{path}/{file}'))
    
    return meshes

path = 'E:\work\kitti360\code/net\experiments\sphere/result'
data_path = 'E:\work\kitti360\code\processed/vegetation/trans'
dirs = os.listdir(path)

for dir in dirs:
    grids = os.listdir(f'{path}/{dir}')

    for grid in grids:
        pcd = npy2pcd(f'{data_path}/{dir}/{grid}/all.npy')

        o3d.visualization.draw_geometries([pcd])

        gt_meshes = get_meshes(f'{path}/{dir}/{grid}/gt')
        meshes = get_meshes(f'{path}/{dir}/{grid}/pre')

        o3d.visualization.draw_geometries([pcd] + gt_meshes)
        o3d.visualization.draw_geometries([pcd] + meshes)
        # quit()