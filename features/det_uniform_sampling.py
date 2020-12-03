from features import detector as det

import os

class UniformSamplingKeypointDetector(det.KeypointDetector):

    """
    Class to detect features uniform sampled
    """

    def __init__(self, *args, **kwargs):
        super(UniformSamplingKeypointDetector, self).__init__(*args, **kwargs)

    def detect_keypoints(self, cloud):
        return cloud.voxel_down_sample(voxel_size=self.radius)
