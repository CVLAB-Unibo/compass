import open3d as o3d
import numpy as np
from abc import ABC, abstractmethod

class KeypointDetector(ABC):
    """
    Abstract keypoint detector class
    """

    def __init__(self, radius):
        self.radius = radius

    @abstractmethod
    def detect_keypoints(self, cloud):
        pass

    @staticmethod
    def compute_keypoints_indices(cloud, keypoints):
        indices = []
        kdtree = o3d.geometry.KDTreeFlann(cloud)

        for j in range(0, np.array(keypoints.points).shape[0]):
            [k, idx, _] = kdtree.search_knn_vector_3d(keypoints.points[j], 1)

            indices.append(idx)

        return indices

    def __call__(self, cloud):

        self.keypoints = o3d.geometry.PointCloud()
        self.keypoints = self.detect_keypoints(cloud)

        self.indices = self.compute_keypoints_indices(cloud, self.keypoints)

        return self.keypoints, np.asarray(self.indices)

    def __repr__(self):
        return self.__class__.__name__ + ' Keypoint detector = {0} '.format(self.radius)

