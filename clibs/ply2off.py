import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh('std.ply')

vertices = np.array(mesh.vertices)

mesh = mesh.simplify_vertex_clustering(0.00001)

o3d.io.write_triangle_mesh('std.ply', mesh)