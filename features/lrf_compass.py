from features import lrf_estimator as lrfe

from dataloader import dataset_localpcd_benchmark as ds
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
        self.layer_s2.to(device)

        # Load local reference frame layer
        self.layer_lrf = nll.LrfLayer(bandwidths=self.lrf_bandwidths[1:],
                                 features=self.lrf_features[1:],
                                 use_equatorial_grid=self.use_equatorial_grid,
                                 softmax_temp=self.softmax_temp)

        utor.load_models_from_ckp(self.path_lrf, self.layer_lrf)

        self.layer_lrf = self.layer_lrf.eval()
        self.layer_lrf.to(device)

    def __call__(self, cloud, indices, return_features_map=False):
        self.dataset = ds.LRFBenchmarkLPCDDataset(cloud, indices, self.radius_support, self.input_transformation)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=self.size_batch,
                                                   shuffle=False,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   drop_last=False)


        lrfs = np.empty((len(indices), 3, 3), dtype='float64')
        clouds = np.empty((len(indices), self.num_points, 3), dtype='float64')
        if (return_features_map):
            features_map = np.empty((len(indices), self.lrf_features[-1], self.lrf_bandwidths[-1]*2, self.lrf_bandwidths[-1]*2, self.lrf_bandwidths[-1]*2))
        else:
            features_map = None

        for i_c, data in enumerate(self.dataloader):
            points, signals, indices_kp = data
            signals = signals.to(self.device)

            with torch.no_grad():

                batch_so3_signals = self.layer_s2(signals)

                lrf_features_map, batch_lrfs = self.layer_lrf(batch_so3_signals)
                batch_lrfs = batch_lrfs.data.cpu().numpy()
                lrf_features_map = lrf_features_map.cpu().numpy()

            lrfs[i_c*self.size_batch : min((i_c+1)*self.size_batch, len(indices))] = batch_lrfs
            clouds[i_c*self.size_batch : min((i_c+1)*self.size_batch, len(indices))] = points
            if (return_features_map):
                features_map[i_c*self.size_batch : min((i_c+1)*self.size_batch, len(indices))] = lrf_features_map

            #torch.cuda.empty_cache() # When commented it increases performances

            # Avoid useless progress bars
            if (len(indices) > 1):
                progress_bar.print_progress(i_c+1, len(self.dataloader))

        
        return lrfs, clouds, features_map

    def __repr__(self):
        return self.__class__.__name__ + ' rad={0}'.format(self.radius_support)
