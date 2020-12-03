import open3d as o3d
from spherical_voxel import spherical_voxel as sv
from lie_learn.spaces import S2
from typing import Tuple
import numpy as np
import math
import copy
import torch


def get_rotation_matrix(alfa, beta, gamma, hom_coord=False):
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, -np.sin(a), 0],
                         [0, 1, 0, 0],
                         [np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(gamma).dot(y(beta)).dot(z(alfa))  # pylint: disable=E1101

    if hom_coord:
        return r
    else:
        return r[:3, :3]


def get_random_rotation_matrix(hom_coord=False):

    alfa = np.random.rand() * 2 * np.pi
    beta = np.random.rand() * 2 - 1
    gamma = np.random.rand() * 2 * np.pi

    mat = get_rotation_matrix(alfa, np.arccos(beta), gamma, hom_coord)

    return mat


def spherical_voxel_optimized(points: np.ndarray, size_bandwidth: int, size_radial_divisions: int,
                              radius_support: float, do_random_sampling: bool, num_random_points: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Compute spherical voxel using the C++ code.

    Compute Spherical Voxel signal as defined in:
    Pointwise Rotation-Invariant Network withAdaptive Sampling and 3D Spherical Voxel Convolution.
    Yang You, Yujing Lou, Qi Liu, Yu-Wing Tai, Weiming Wang, Lizhuang Ma and Cewu Lu.
    AAAI 2020.

    :param points: the points to convert.
    :param size_bandwidth: alpha and beta bandwidth.
    :param size_radial_divisions: the number of bins along radial dimension.
    :param radius_support: the radius used to compute the points in the support.
    :param do_random_sampling: if true a subset of random points will be used to compute the spherical voxel.
    :param num_random_points: the number of points to keep if do_random_sampling is true.

    :return: A tuple containing:
        The spherical voxel, shape(size_radial_divisions, 2 * size_bandwidth, 2 * size_bandwidth).
        The points used to compute the signal normalized according the the farthest point.
    """
    if do_random_sampling:
        min_limit = 1 if points.shape[0] > 1 else 0
        indices_random = np.random.randint(min_limit, points.shape[0], num_random_points)
        points = points[indices_random]

    pts_norm = np.linalg.norm(points, axis=1)
    # Scale points to fit unit sphere
    pts_normed = points / pts_norm[:, None]
    pts_normed = np.clip(pts_normed, -1, 1)

    pts_s2_coord = S2.change_coordinates(pts_normed, p_from='C', p_to='S')
    # Convert to spherical voxel indices
    pts_s2_coord[:, 0] *= 2 * size_bandwidth / np.pi  # [0, pi]
    pts_s2_coord[:, 1] *= size_bandwidth / np.pi
    pts_s2_coord[:, 1][pts_s2_coord[:, 1] < 0] += 2 * size_bandwidth

    # Adaptive sampling factor
    daas_weights = np.sin(np.pi * (2 * np.arange(2 * size_bandwidth) + 1) / 4 / size_bandwidth).astype(np.float32)
    voxel = np.asarray(sv.compute(pts_on_s2=pts_s2_coord,
                                  pts_norm=pts_norm,
                                  size_bandwidth=size_bandwidth,
                                  size_radial_divisions=size_radial_divisions,
                                  radius_support=radius_support,
                                  daas_weights=daas_weights))
    pts_normed = points / np.max(pts_norm)
    return voxel.astype(np.float32), pts_normed.astype(np.float32)


def lrf_repeatability(lrfs_src, lrfs_trg, mat_from_src_to_trg, th_cosine=0.97):
    """
    Compute local reference frame repeatability considering the cosine of angles between x and z axes.
    Two frames are repeatable if the cosines between x and z axes are greater than th_cosine
    :param lrfs_src: local reference frames on src (axes on COLUMNS)
    :param lrfs_trg: local reference frames on trg (axes on COLUMNS)
    :param mat_from_src_to_trg: matrices from src to trg
    :param th_cosine: threshold on cosine to consider the axes repeatable, default 0.97
    :return: an array with 0 and 1 indicating whether the lrf is repeatable or not
    """

    lrfs_src_in_trg = mat_from_src_to_trg @ lrfs_src

    muls = np.multiply(lrfs_src_in_trg, lrfs_trg)

    dots = np.sum(muls, axis=1)

    dots_x = dots[:, 0]
    dots_z = dots[:, 2]

    positives_x = dots_x >= th_cosine
    positives_z = dots_z >= th_cosine

    res = positives_x * positives_z

    return 0 + res


def get_dimensions(cloud):
    min = np.min(cloud, axis=0)
    max = np.max(cloud, axis=0)

    return (max - min)


def get_max_radius(cloud):
    bb_cloud = get_dimensions(cloud)
    return np.max(bb_cloud)*0.5


def get_overlapping_area(cloud_a, cloud_b, mat_pose_a, mat_pose_b, min_euclidean_distance_to_consider_kp_in_fragment):

    cloud_a_in_gt = copy.deepcopy(cloud_a)
    cloud_b_in_gt = copy.deepcopy(cloud_b)

    cloud_a_in_gt.transform(mat_pose_a)
    cloud_b_in_gt.transform(mat_pose_b)

    kdtree_b = o3d.geometry.KDTreeFlann(cloud_b_in_gt)
    kdtree_a = o3d.geometry.KDTreeFlann(cloud_a_in_gt)

    dists = []

    for i in range(0, np.array(cloud_a_in_gt.points).shape[0]):
        [_, idx_a, dist_a] = kdtree_b.search_knn_vector_3d(cloud_a_in_gt.points[i], 1)
        [_, idx_b, _] = kdtree_a.search_knn_vector_3d(cloud_b_in_gt.points[idx_a[0]], 1)

        if idx_b[0] == i and dist_a[0] < min_euclidean_distance_to_consider_kp_in_fragment:
            dists.append(dist_a[0])

    overlap_area = np.array(dists).size / np.array(cloud_a_in_gt.points).shape[0]

    return overlap_area


def compute_nearest_search_intersection(cloud_src, cloud_trg, min_distance):

    indices_trg = []
    indices_src = []

    kdtree = o3d.geometry.KDTreeFlann(cloud_trg)

    for idx_src, point_src in enumerate(cloud_src.points):
        [k, idx, distance] = kdtree.search_knn_vector_3d(point_src, 1)

        if np.sqrt(distance[0]) <= min_distance:
            indices_trg.append(idx[0])
            indices_src.append(idx_src)

    cloud_points_on_trg = o3d.geometry.PointCloud()
    cloud_points_on_trg.points = o3d.utility.Vector3dVector(np.asarray(cloud_trg.points)[indices_trg])

    return cloud_points_on_trg, np.asarray(indices_trg), np.asarray(indices_src)




# What follows is taken from https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# and is subject to the following conditions

# Copyright (c) 2006-2019, Christoph Gohlke
# Copyright (c) 2006-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0] # 1e-8 was apparently too strict (Federico Stella)
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point
