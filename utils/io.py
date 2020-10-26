from plyfile import PlyData
import open3d as o3d

import numpy as np
import h5py


def read_ply(name_file):

    ply = PlyData.read(name_file)
    np_array = np.array([[x, y, z] for x, y, z in ply['vertex'].data]).astype(np.float64)

    return np_array


def read_ply_to_cloud_open3d(name_file):

    ply = PlyData.read(name_file)
    np_array = np.array([[x, y, z] for x, y, z in ply['vertex'].data]).astype(np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np_array)

    return cloud

def read_mesh_to_cloud(name_file):
    mesh = o3d.io.read_triangle_mesh(name_file)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(mesh.vertices)

    return cloud


def save_pcd(name_file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(name_file, pcd)


def string_to_float(list, dtype=np.float32):
    try:
        list_float = [float(x) for x in list]
    except ValueError:
        print("Failed to convert {}".format(list))

    return np.asarray(list_float).astype(dtype)


def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
