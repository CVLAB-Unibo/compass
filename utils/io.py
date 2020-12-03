import open3d as o3d
from plyfile import PlyData

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

def get_pair_and_matrix_3dmatch(file, delimiter="\t", type=np.float32):

    dict = {}

    with open(file, "r") as file_gt:
        lines = file_gt.readlines()

        for i in range(0, len(lines), 5):
            key = "{}-{}".format(lines[i + 0].split(delimiter)[0].strip(), lines[i + 0].split(delimiter)[1].strip())
            line_matrix_row0 = string_to_float(str(lines[i + 1]).strip().split(delimiter), type)
            line_matrix_row1 = string_to_float(str(lines[i + 2]).strip().split(delimiter), type)
            line_matrix_row2 = string_to_float(str(lines[i + 3]).strip().split(delimiter), type)
            line_matrix_row3 = string_to_float(str(lines[i + 4]).strip().split(delimiter), type)

            dict[key] = np.stack([line_matrix_row0, line_matrix_row1, line_matrix_row2, line_matrix_row3])

    return dict


def get_overlapping_areas_3dmatch(file, delimiter="\t", type=np.float32):
    return {}


def get_pairs_and_matrix_stanford(file, delimiter="\t", type=np.float32):

    dict_view_gt = {}

    with open(file, "r") as file_gt:
        lines = file_gt.readlines()
        num_views = int(lines[0].strip().split(":")[-1])

        lines = lines[1:]

        for i in range(0, len(lines), 5):
            name_cloud = lines[i + 0].strip().split(".")[0]
            line_matrix_row0 = string_to_float(str(lines[i + 1]).strip().split(delimiter), type)
            line_matrix_row1 = string_to_float(str(lines[i + 2]).strip().split(delimiter), type)
            line_matrix_row2 = string_to_float(str(lines[i + 3]).strip().split(delimiter), type)
            line_matrix_row3 = string_to_float(str(lines[i + 4]).strip().split(delimiter), type)

            dict_view_gt[name_cloud] = np.stack([line_matrix_row0, line_matrix_row1, line_matrix_row2, line_matrix_row3])

    dict_pairs = {}
    list_keys = list(dict_view_gt.keys())
    for i in range(num_views):
        for j in range(i, num_views):
            if i != j:
                name_src = list_keys[i]
                name_trg = list_keys[j]

                mat_src = dict_view_gt[name_src]
                mat_trg = dict_view_gt[name_trg]

                mat_trg_to_src = np.linalg.inv(mat_src) @ mat_trg
                dict_pairs["{}-{}".format(name_src, name_trg)] = mat_trg_to_src

    return dict_pairs


def get_overlapping_areas_stanford(file):
    dict_overlap = {}

    with open(file, "r") as file_gt:
        lines = file_gt.readlines()

        lines = lines[5:]

        for i in range(0, len(lines)):
            name_cloud_src = lines[i].strip().split(" ")[3]
            name_cloud_trg = lines[i].strip().split(" ")[4]
            overlap = float(str(lines[i].strip().split(" ")[5]))

            dict_overlap["{}-{}".format(name_cloud_src, name_cloud_trg)] = overlap

    return dict_overlap
