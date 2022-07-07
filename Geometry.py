import numpy as np
import os
import open3d as o3d

def npy2pcd(path):
    pts = np.load(path)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    return pcd

def visualize_sample(pts, R, T, out, args):
    pts, R, T, out = np.array(pts), np.array(R), np.array(T), np.array(out)

    bbox = o3d.geometry.TriangleMesh.create_box(20, 20, 20).translate((-10, -10, -10))
    bbox_line = o3d.geometry.LineSet.create_from_triangle_mesh(bbox)

    mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_path, 'std.ply'))
    points = np.array(mesh.vertices) @ R + T
    mesh.vertices = o3d.utility.Vector3dVector(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=T)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=out[-1])
    sphere = sphere.translate((out[:3]))

    o3d.visualization.draw_geometries([pcd, FOR, bbox_line])
    o3d.visualization.draw_geometries([mesh, pcd, FOR, bbox_line])
    o3d.visualization.draw_geometries([mesh, pcd, sphere, bbox_line])

def goback(mesh, scales, trans):
    points = np.array(mesh.vertices) / scales + trans
    mesh.vertices = o3d.utility.Vector3dVector(points)

def test_sample(gt_sphere, sphere, scales, trans, pcd_path, args):
    gt_sphere, sphere, scales, trans = np.array(gt_sphere), np.array(sphere), np.array(scales), np.array(trans)

    gt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=gt_sphere[-1]).translate((gt_sphere[:3]))
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere[-1]).translate((sphere[:3]))

    pcd = npy2pcd(os.path.join(args.data_path, pcd_path))

    goback(gt_mesh, scales, trans)
    goback(mesh, scales, trans)

    return gt_mesh, mesh
    # o3d.visualization.draw_geometries([gt_mesh, pcd])
    # o3d.visualization.draw_geometries([mesh, pcd])
    # gt_points = np.array(gt_mesh.vertices) / scales + trans
    # points = np.array(mesh.vertices) / scales + trans

    # gt_mesh.vertices = o3d.utility.Vector3dVector(gt_points)
    # mesh.vertices = o3d.utility.Vector3dVector(points)


    # quit()