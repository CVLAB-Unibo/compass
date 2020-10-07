from utils import geometry as ug


class ConvertToSphericalVoxel():

    """
    Convert point cloud to spherical voxel [beta = 2 * bandwidth,  alfa = 2 * bandwidth, num_radial_division].
    Alfa in [0, 2pi], Beta in [0, pi]
    """

    def __init__(self, bandwidth, radius_support, num_radial_division, num_points, random_sampling):

        self.bandwidth = bandwidth
        self.radius_support = radius_support
        self.num_radial_division = num_radial_division
        self.num_points = num_points
        self.random_sampling = random_sampling

    def __call__(self, point_cloud):

        features, pts_normed = ug.spherical_voxel_optimized(points=point_cloud,
                                                size_bandwidth=self.bandwidth,
                                                size_radial_divisions=self.num_radial_division,
                                                radius_support=self.radius_support,
                                                do_random_sampling=self.random_sampling,
                                                num_random_points=self.num_points)

        return features, pts_normed

    def __repr__(self):
        return self.__class__.__name__ + 'Bandwidth = {0} ' \
                                         '- Radius Support = {1} ' \
                                         '- Num Radial Division = {2} ' \
                                         '- Random Sampling = {3} ' \
                                         '- Number of points = {4} '.format(self.bandwidth,
                                                                            self.radius_support,
                                                                            self.num_radial_division,
                                                                            self.random_sampling,
                                                                            self.num_points)

