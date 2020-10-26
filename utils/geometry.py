'''

Geometry utils

Author: Riccardo Spezialetti
Mail: riccardo.spezialetti@unibo.it

'''
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
    