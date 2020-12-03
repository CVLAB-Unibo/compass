from features import lrf_estimator as lrfe

from dataloader import dataset_localpcd as ds
from dataloader import spherical_voxel as sv

from models import network_s2_layer as nsl
from models import network_lrf_layer as nll

from utils import torch as utor
from utils import progress_bar

import numpy as np

from s2cnn import so3_rotation
import torch


class CompassEstimator(lrfe.LRFEstimator):
    """
    Self orienting canonical LRF estimator
    """

    def __init__(self, path_s2, path_lrf, device, num_points, radius_support, lrf_bandwidths, lrf_features, use_equatorial_grid, softmax_temp, num_workers, size_batch, *args, **kwargs):
        super(CompassEstimator, self).__init__(*args, **kwargs)

        self.path_s2 = path_s2
        self.path_lrf = path_lrf
        self.device = device
        self.num_points = num_points
        self.radius_support = radius_support
        self.lrf_features = lrf_features
        self.lrf_bandwidths = lrf_bandwidths
        self.use_equatorial_grid = use_equatorial_grid
        self.softmax_temp = softmax_temp
        self.num_workers = num_workers
        self.size_batch = size_batch

        self.input_transformation = sv.ConvertToSphericalVoxel(bandwidth=self.lrf_bandwidths[0],
                                                        radius_support=self.radius_support,
                                                        num_radial_division=self.lrf_features[0],
                                                        num_points=self.num_points,
                                                        random_sampling=True)
        
        # Load S2 layer
        self.layer_s2 = nsl.S2Layer(bandwidths=self.lrf_bandwidths[0:2],
                                features=self.lrf_features[0:2],
                                use_equatorial_grid = self.use_equatorial_grid)

        utor.load_models_from_ckp(self.path_s2, self.layer_s2)

        self.layer_s2 = self.layer_s2.eval()
        self.layer_s2.to(self.device)

        # Load local reference frame layer
        self.layer_lrf = nll.LrfLayer(bandwidths=self.lrf_bandwidths[1:],
                                 features=self.lrf_features[1:],
                                 use_equatorial_grid=self.use_equatorial_grid,
                                 softmax_temp=self.softmax_temp)

        utor.load_models_from_ckp(self.path_lrf, self.layer_lrf)

        self.layer_lrf = self.layer_lrf.eval()
        self.layer_lrf.to(self.device)

    def __call__(self, cloud):
        self.input_transformation.radius_support = self.radius_support

        signal, _ = self.input_transformation(cloud)

        signal = torch.Tensor(signal).unsqueeze(0)

        with torch.no_grad():
            signal = signal.to(self.device)
            batch_so3_signals = self.layer_s2(signal)
            batch_so3_signals = batch_so3_signals.to(self.device)
            lrf_features_map, batch_lrfs = self.layer_lrf(batch_so3_signals)
            lrfs = batch_lrfs.data.cpu().numpy()

        return lrfs

    def __repr__(self):
        return self.__class__.__name__ + ' rad={0}'.format(self.radius_support)
