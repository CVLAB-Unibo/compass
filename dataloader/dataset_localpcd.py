# pylint: disable=E1101,R,C

import open3d as o3d

import csv
import glob
import math
import os
import torch.utils.data
import time
import random

import numpy as np

from utils import io as uio
from utils import geometry as ug
from utils import progress_bar

class LocalPointCloudDataset(torch.utils.data.Dataset):

    """
    PyTorch Dataset component to extract the local support of a given feature point.
    Given a root folder and a file listing the name of subfolders, the keypoints would be uniform sampled from each
    pointcloud file (with given extension) in each subfolder.
    The extracted local support is saved as npy file to serve as cached during train.
    """

    def __init__(self,
                 path_root,
                 path_npy,
                 file_list_folders,
                 extension,
                 limit,
                 min_nn,
                 radius,
                 size_leaf_keypoints,
                 size_leaf_ss,
                 augmentation,
                 removal_augmentation,
                 transform,
                 dataset="3DMatch"):
        """

        :param path_root: path to the root folder
        :param path_npy: path to npy file where store the cached local support for each keypoints
        :param file_list_folders: file containing the subfolder of pathroot with the point cloud files
        :param dataset: 3DMatch or StanfordViews
        :param extension: pointcloud files extension
        :param limit: if is not zero limit the dataset to use only limit number of files
        :param min_nn: number of minimum points in the support to consider the feature point in the dataset
        :param radius: radius for local support
        :param size_leaf_keypoints: leaf for uniform sampling keypoint detector
        :param size_leaf_ss: leaf for uniform sampling of the input pointcloud
        :param augmentation: add random rotation to input
        :param removal_augmentation: add removal augmentation to the input
        :param transform: transformation for the input data
        """

        self.augmentation = augmentation
        self.removal_augmentation = removal_augmentation
        self.dataset = dataset
        self.extension = extension
        self.limit = limit
        self.min_nn = min_nn
        self.num_samples = 0
        self.num_split = 300
        self.root = os.path.expanduser(path_root)
        self.file_list_folders = file_list_folders
        self.files = []
        self.path_npy = path_npy
        self.radius = radius
        self.leaf_keypoints = size_leaf_keypoints
        self.size_leaf = size_leaf_ss
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found in {}.'.format(path_root))

    def __getitem__(self, index):

        name = self.get_single_sample_path(index)
        pts = np.load(name)

        # Points Random Rotated
        pts_rnd = np.copy(pts)

        mat_rnd = ug.get_random_rotation_matrix()

        pts_rnd = pts_rnd @ mat_rnd.T

        # Apply random rotation#
        if self.augmentation:
            mat_augmentation = ug.get_random_rotation_matrix()

            mat_rnd = mat_rnd @ mat_augmentation.T
            pts = pts @ mat_augmentation.T

        if self.removal_augmentation: #Applied to the target copy only
            rand = random.randint(1,10)
            if rand <= 5: # 50% probability of not occluding anything
                pass
            else:
                #print("Point-removal augmentation")
                #Compute distances from the keypoint
                dist = np.linalg.norm(pts_rnd, axis=1)

                #Divide the points in 3 spherical shells by dividing the support radius in 3 equal segments
                dist = dist / self.radius * 3 #Rescaling to [0,3]
                dist = np.floor(dist) #Converting to integers from 0 to 2, representing the 3 different shells
                
                # Randomly select one shell
                shell = np.empty(1)
                if rand <= 6: # 20% probability of taking the central shell
                    shell = np.where(dist == 1)[0]
                    #print("On shell 1")
                else: # 80% probability of taking the outer shell  
                    shell = np.where(dist == 2)[0]
                    #print("On shell 2")

                # Randomly select one point of the shell
                if len(shell) > 0:
                    random_index = random.randint(0,len(shell)-1)
                    random_point_index = shell[random_index]

                    # Randomly select the number of nearest neighbors to delete [5,20]% of total points
                    random_k = int(len(dist) * (random.randint(5,20) / 100))
                    #print("Deleted " + str(random_k) + " points (" + str(random_k/len(dist)*100) + "%% of the total)")

                    # Build KNN tree
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(pts_rnd)
                    kdtree = o3d.geometry.KDTreeFlann(cloud)
                    [_, indices, _] = kdtree.search_knn_vector_3d(pts_rnd[random_point_index], random_k)

                    # Delete points
                    pts_rnd = np.delete(pts_rnd, np.array(indices), axis=0)

        if self.transform is not None and type(self.transform).__name__ == "ConvertToSphericalVoxel":
            if (self.dataset == "ShapeNet" or self.dataset == "ModelNet"):
                self.transform.radius_support = ug.get_max_radius(pts)

            signal, points_normed = self.transform(pts)
            signal_rnd, points_normed_rnd = self.transform(pts_rnd)

            return signal, points_normed, signal_rnd, points_normed_rnd, mat_rnd, name.split('/')[-1]
        else:
            if (self.dataset == "ShapeNet" or self.dataset == "ModelNet"):
                self.transform.radius_support = ug.get_max_radius(pts)
            pts = self.transform(pts)
            return pts, name.split('/')[-1]

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return self.__class__.__name__ + ' Path = {0} - Support = {1} - Leaf features = {2} - Leaf SS = {3}'.format(self.root,
                                                                                                                     self.radius,
                                                                                                                     self.leaf_keypoints,
                                                                                                                     self.size_leaf)

    def _check_exists(self):

        with open(self.file_list_folders) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')

            for row in csv_reader:
                if(self.dataset == "ModelNet"):
                    illegal_name = os.path.join(self.root, str(row[0]), row[0])
                    filenames = glob.glob(os.path.join(self.root, str(row[0])))
                    filenames = [filename for filename in filenames if filename != illegal_name]
                    self.files += filenames
                else:
                    illegal_name = os.path.join(self.root, str(row[0]), row[0] + "." + self.extension)
                    filenames = glob.glob(os.path.join(self.root, str(row[0]), "*." + self.extension))
                    filenames = [filename for filename in filenames if filename != illegal_name]
                    self.files += filenames

        if self.limit is not 0:
            idx = min(self.limit, len(self.files))
            self.files = self.files[:idx]

        return len(self.files) > 0

    def store_npy(self):

        # Build the dataset
        time_start = time.perf_counter()

        self.save_keypoints_support()

        return time.perf_counter() - time_start

    def get_single_sample_path(self, num_sample):

        if num_sample % self.num_split == 0:
            num_folder = math.ceil(num_sample / self.num_split)
        else:
            num_folder = max(math.ceil(num_sample / self.num_split) - 1, 0)

        name_file = 's_%d.npy' % num_sample

        return os.path.join(self.path_npy, str(num_folder), name_file)


    def save_keypoints_support(self):

        with open(os.path.join(self.path_npy, 'clouds_to_npz.csv'), 'w') as csvfile:

            file_writer = csv.writer(csvfile,
                                     delimiter=';',
                                     quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)

            file_writer.writerow(['Scene', 'Cloud', 'P_x', 'P_y', 'P_z', 'Folder', 'Npz'])

            num_folder = 0

            for i, f in enumerate(self.files):

                if (self.dataset == "ShapeNet"):
                    cloud = uio.read_ply_to_cloud_open3d(f)
                    size_cloud_keypoints = 1
                elif (self.dataset == "ModelNet"):
                    data, _ = uio.load_h5(f)
                    size_cloud_keypoints = data.shape[0]
                    cloud = o3d.geometry.PointCloud()
                else:
                    if (self.dataset == "3DMatch"):
                        cloud = uio.read_ply_to_cloud_open3d(f)
                    if (self.dataset == "StanfordViews"):
                        cloud = uio.read_mesh_to_cloud(f)
                    cloud_keypoints = cloud.voxel_down_sample(voxel_size=self.leaf_keypoints)
                    size_cloud_keypoints = np.array(cloud_keypoints.points).shape[0]
                    cloud_keypoints.paint_uniform_color([0, 0, 1])

                if (self.size_leaf > 0):
                    cloud_down = cloud.voxel_down_sample(self.size_leaf)
                else:
                    cloud_down = cloud

                if (self.dataset != "ShapeNet" and self.dataset != "ModelNet"):
                    cloud_down.paint_uniform_color([0, 0, 1])
                    kdtree = o3d.geometry.KDTreeFlann(cloud_down)

                for j in range(0, size_cloud_keypoints):

                    if (self.dataset == "ShapeNet"):
                        cloud_nn = np.asarray(cloud_down.points)
                    elif (self.dataset == "ModelNet"):
                        cloud_nn = np.asarray(data[j], dtype=np.float32)
                    else:
                        [k, idx, _] = kdtree.search_radius_vector_3d(cloud_keypoints.points[j], self.radius)
                        cloud_nn = np.asarray(np.asarray(cloud_down.points)[idx[1:], :]) - np.asarray(
                            cloud_keypoints.points[j])

                    if cloud_nn.shape[0] >= self.min_nn:

                        if self.num_samples % self.num_split == 0:

                            name_folder = str(num_folder)

                            path_folder = os.path.join(self.path_npy, name_folder)
                            num_folder += 1

                            if not os.path.exists(path_folder) and not os.path.isdir(path_folder):
                                os.mkdir(path_folder)

                        name_file = 's_%d.npy' % self.num_samples
                        np.save(os.path.join(path_folder, name_file), cloud_nn)

                        if (self.dataset == "ShapeNet" or self.dataset == "ModelNet"):
                            file_writer.writerow([f.split('/')[-2],
                                                  f.split('/')[-1],
                                                  name_folder,
                                                  name_file])
                        else:
                            file_writer.writerow([f.split('/')[-2],
                                                  f.split('/')[-1],
                                                  cloud_keypoints.points[j][0],
                                                  cloud_keypoints.points[j][1],
                                                  cloud_keypoints.points[j][2],
                                                  name_folder,
                                                  name_file])



                        self.num_samples += 1

                progress_bar.print_progress(i+1, len(self.files))
