import numpy as np
import os
import open3d as o3d
from regex import F

from Geometry import npy2pcd

def get_meshes(path):
    files = os.listdir(path)

    meshes = []
    for file in files:
        if file.endswith('ply'):
            meshes.append(o3d.io.read_triangle_mesh(f'{path}/{file}'))
    
    return meshes

path = 'E:\work\kitti360\code/net\experiments\sphere/result'
data_path = 'E:\work\kitti360\code\processed/vegetation/trans'
dirs = os.listdir(path)

work_dir = 'E:\work\kitti360\code/net\experiments\sphere'

mesh_dir = 'E:\work\kitti360\kitti360Scripts\kitti360scripts\custom/all_bboxes'

for dir in dirs:
    grids = os.listdir(f'{path}/{dir}')

    for grid in grids:
        pcd = npy2pcd(f'{data_path}/{dir}/{grid}/all.npy')

        # o3d.visualization.draw_geometries([pcd])
        origin_meshes = get_meshes(f'{mesh_dir}/{dir}/21/{grid}')
        gt_meshes = get_meshes(f'{path}/{dir}/{grid}/gt')
        meshes = get_meshes(f'{path}/{dir}/{grid}/pre')

        # o3d.visualization.draw_geometries([pcd] + gt_meshes)
        # o3d.visualization.draw_geometries([pcd] + meshes)

        origin_mesh = None
        for mesh in origin_meshes:
            if len(mesh.vertices) == 63:
                if origin_mesh is None:
                    origin_mesh = mesh
                else:
                    origin_mesh += mesh

        for i in range(1, len(meshes)):
            gt_meshes[0] += gt_meshes[i]
            meshes[0] += meshes[i]

        os.makedirs(f'{work_dir}/samples/{dir}/{grid}', exist_ok=True)
        o3d.io.write_point_cloud(f'{work_dir}/samples/{dir}/{grid}/pcd.ply', pcd)
        o3d.io.write_triangle_mesh(f'{work_dir}/samples/{dir}/{grid}/gt.ply', gt_meshes[0])
        o3d.io.write_triangle_mesh(f'{work_dir}/samples/{dir}/{grid}/pre.ply', meshes[0])
        o3d.io.write_triangle_mesh(f'{work_dir}/samples/{dir}/{grid}/origin.ply', origin_mesh)
        # quit()