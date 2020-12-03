# pylint: disable=E1101,R,C

import open3d as o3d

import torch.utils.data

import numpy as np

from utils import io as uio
from utils import geometry as ug
from features import det_uniform_sampling as kpus


class LRFBenchmarkLPCDDataset(torch.utils.data.Dataset):

    """
    PyTorch Dataset component to extract the local support of a given feature point.
    Initialized with a pair of clouds and a ground truth rotation matrix between them,
    the keypoints are uniformly sampled, local support extract and spherical voxelization computed.
    """

    def __init__(self,
                 cloud,
                 indices_kp,
                 support_radius,
                 input_transformation):
        """

        :param cloud: point cloud
        :param indices_kp: arrays of keypoint indices
        :param support_radius: radius for local support
        :param input_transformation: transformation to apply to local support
        """
        self.cloud = cloud
        self.indices_kp = indices_kp
        self.support_radius = support_radius
        self.input_transformation = input_transformation

        self.kdtree = o3d.geometry.KDTreeFlann(self.cloud)

    def __getitem__(self, index):
        index_kp = self.indices_kp[index]
        points, signal = self._transform(index_kp)

        return points, signal, index_kp

    def __len__(self):
        return len(self.indices_kp)

    def __repr__(self):
        return self.__class__.__name__ + 'Radius Support = {}'.format(self.support_radius)

    def _transform(self, point_index): 
        feature_point = self.cloud.points[point_index]

        [_, idx, _] = self.kdtree.search_radius_vector_3d(feature_point, self.support_radius)

        if np.asarray(idx).shape[0] == 1:
            # raise e.EmptySupportException('The support for point {} is empty'.format(idx_keypoint))
            # signal = torch.zeros(self.input_transformation.num_radial_division,
            #                         self.input_transformation.bandwidth * 2,
            #                         self.input_transformation.bandwidth * 2)
            # pts_normed = torch.rand(self.input_transformation.num_points, 3)
            # ^ Why is it with torch?
            signal = torch.zeros(self.input_transformation.num_radial_division,
                                    self.input_transformation.bandwidth * 2,
                                    self.input_transformation.bandwidth * 2).cpu().numpy()
            pts_normed = torch.rand(self.input_transformation.num_points, 3).cpu().numpy()
        else:
            cloud_nn = np.asarray(np.asarray(self.cloud.points)[idx[1:], :]) - np.asarray(feature_point)
            signal, pts_normed = self.input_transformation(cloud_nn)

        return pts_normed, signal
