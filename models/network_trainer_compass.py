from dataloader import dataset_localpcd
from dataloader import spherical_voxel as sv

from models import loss as l
from models import network_trainer as nt
from models import network_s2_layer as nsl
from models import network_lrf_layer as nll
from models import soft_argmax as sfa

from utils import io as uio
from utils import geometry as ug
from utils import file_system as ufs
from utils import torch as utor
from utils import train as utr


import numpy as np
import shutil
from termcolor import colored
import torch
import torchnet.meter as tnt
import os


class CompassNetworkTrainer(nt.NetworkTrainer):

    def __init__(self, *args, **kwargs):
        super(CompassNetworkTrainer, self).__init__(*args, **kwargs)

        self.layer_s2 = None
        self.layer_lrf = None
        self.softmax_temp = 1.0
        self.use_equatorial_grid = 0

        self.th_cosine = 0.97

        self.criterion_theta = None
        self.name_losses = ['net', 'theta', 'lrf_rep']

        self.histogram_theta_train = []
        self.histogram_theta_validation = []

    def init_output_dirs(self):

        self.dict_paths['pcd_train'] = os.path.join(self.args.path_log, "pcd_train")
        self.dict_paths['pcd_val'] = os.path.join(self.args.path_log, "pcd_val")
        self.dict_paths['checkpoint'] = os.path.join(self.args.path_log, "checkpoints")
        self.dict_paths['npz_lrfs'] = os.path.join(self.args.path_log, "npz_lrfs")

        for path in self.dict_paths.values():
            ufs.make_dir(path)

        return True

    def init_network(self):
        name_components = ["network_lrf_layer.py"]

        # Load S2 Layer
        self.layer_s2 = nsl.S2Layer(bandwidths=self.args.lrf_bandwidths[0:2],
                                 features=self.args.lrf_features[0:2],
                                 use_equatorial_grid = self.use_equatorial_grid)

        # Load local reference frame layer
        self.layer_lrf = nll.LrfLayer(bandwidths=self.args.lrf_bandwidths[1:],
                                      features=self.args.lrf_features[1:],
                                      softmax_temp=self.softmax_temp,
                                      use_equatorial_grid=self.use_equatorial_grid)

        # Save Network to file
        file_net = open(os.path.join(self.args.path_log, self.args.name_train + "_net.info"), "w")

        utor.print_network_module(self.layer_s2, file_net)
        utor.print_network_module(self.layer_lrf, file_net)

        file_net.close()

        return True

    def init_dataloader_train(self):

        transform = sv.ConvertToSphericalVoxel(bandwidth=self.args.size_bandwidth,
                                               radius_support=self.args.radius_descriptor,
                                               num_radial_division=self.args.size_channels,
                                               num_points=self.args.size_pcd,
                                               random_sampling=True)

        path_npy_train = os.path.join(self.args.path_log, "npy_train")

        
        dataset_train = dataset_localpcd.LocalPointCloudDataset(path_root=self.args.path_ds,
                                                                limit=0, #No limit
                                                                min_nn=300,
                                                                path_npy=path_npy_train,
                                                                file_list_folders=self.args.name_file_folder_train,
                                                                extension=self.args.ext,
                                                                radius=self.args.radius_descriptor,
                                                                size_leaf_keypoints=self.args.leaf_keypoints,
                                                                size_leaf_ss=self.args.leaf_sub_sampling,
                                                                augmentation=True,
                                                                removal_augmentation=bool(self.args.removal_augmentation),
                                                                transform=transform,
                                                                dataset=self.args.name_data_set)


        ufs.make_dir(path_npy_train)
        self.logger.info("Start saving NPZ train dataset {0}".format(path_npy_train))

        _ = dataset_train.store_npy()

        train_loader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=self.args.size_batch,
                                                   shuffle=True,
                                                   num_workers=self.args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True)

        return train_loader

    def init_dataloader_validation(self):

        transform = sv.ConvertToSphericalVoxel(bandwidth=self.args.size_bandwidth,
                                               radius_support=self.args.radius_descriptor,
                                               num_radial_division=self.args.size_channels,
                                               num_points=self.args.size_pcd,
                                               random_sampling=True)

        path_npy_val = os.path.join(self.args.path_log, "npy_val")

        dataset_validation = dataset_localpcd.LocalPointCloudDataset(path_root=self.args.path_ds,
                                                                    limit=0, #No limit
                                                                    min_nn=300,
                                                                    path_npy=path_npy_val,
                                                                    file_list_folders=self.args.name_file_folder_validation,
                                                                    extension=self.args.ext,
                                                                    radius=self.args.radius_descriptor,
                                                                    size_leaf_keypoints=self.args.leaf_keypoints,
                                                                    size_leaf_ss=self.args.leaf_sub_sampling,
                                                                    augmentation=True,
                                                                    removal_augmentation=bool(self.args.removal_augmentation),
                                                                    transform=transform,
                                                                    dataset=self.args.name_data_set)

        ufs.make_dir(path_npy_val)
        self.logger.info("Start saving NPZ validation dataset {0}".format(path_npy_val))
        _ = dataset_validation.store_npy()

        validation_loader = torch.utils.data.DataLoader(dataset_validation,
                                                        batch_size=self.args.size_batch,
                                                        shuffle=True,
                                                        num_workers=self.args.num_workers,
                                                        pin_memory=True,
                                                        drop_last=True)
        return validation_loader

    def init_losses(self):
        self.criterion_theta = l.ThetaBorisovLoss(self.device)

        return True

    def init_optimizer(self):
        # Print summary. The net hasn't been uploaded to GPU yet, so this works for CPU mode only.
        # from torchsummary import summary
        # p1 = list(self.layer_s2.parameters())
        # p2 = list(self.layer_lrf.parameters())
        # summary(self.layer_s2, (4,24,24), batch_size=2)
        # summary(self.layer_lrf, (40,24,24,24), batch_size=2)
        
        params = list(self.layer_s2.parameters()) \
                 + list(self.layer_lrf.parameters())

        self.logger.info("{} parameters in total".format(sum(x.numel() for x in params)))

        # Optimizer
        self.lr = self.args.learning_rate
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        return True

    def init_meters_train(self):

        for loss_name in self.name_losses:
            self.dict_meter_loss_train[loss_name] = tnt.AverageValueMeter()

        return True

    def init_meters_validation(self):

        for loss_name in self.name_losses:
            self.dict_meter_loss_val[loss_name] = tnt.AverageValueMeter()

        return True

    def init_curves_train(self):

        for loss_name in self.name_losses:
            self.dict_curves_loss_train[loss_name] = []

    def init_curves_validation(self):

        for loss_name in self.name_losses:
            self.dict_curves_loss_validation[loss_name] = []
            self.dict_curves_loss_train_per_validation[loss_name] = []

    def load_from_checkpoint(self):

        # Network models
        if utor.load_models_from_ckp(self.args.path_ckp_s2_layer, self.layer_s2):
            self.logger.info("Loaded s2 layer from checkpoint {0}.".format(self.args.path_ckp_s2_layer))

        if utor.load_models_from_ckp(self.args.path_ckp_lrf_layer, self.layer_lrf):
            self.logger.info("Loaded lrf from checkpoint {0}.".format(self.args.path_ckp_lrf_layer))

        self.move_network_to_device()

        # Optimizer
        if self.args.path_ckp_ts is not None:
            dict_ckp_train_stuff = torch.load(self.args.path_ckp_ts)
            self.epoch_start = dict_ckp_train_stuff['epochs']
            self.lr = dict_ckp_train_stuff['lr']

            for p in self.optimizer.param_groups:
                p['lr'] = self.lr

            self.optimizer.load_state_dict(dict_ckp_train_stuff['optimizer_state_dict'])

            for key in dict_ckp_train_stuff:
                self.logger.info("Loaded:{} from {}".format(key, self.args.path_ckp_ts))

    def shutdown(self):
        pass

    def move_network_to_device(self):

        self.layer_s2.to(self.device)
        self.layer_lrf.to(self.device)

    def put_network_in_train(self):

        self.layer_s2.train()
        self.layer_lrf.train()

    def put_network_in_eval(self):
        self.layer_s2.eval()
        self.layer_lrf.eval()

    def decay_learning_rate(self):

        self.logger.info("learning rate = {}".format(self.lr))

        for p in self.optimizer.param_groups:
            p['lr'] = self.lr

    def save_checkpoint(self):
        torch.save({'epochs': self.epoch_current, 'optimizer_state_dict': self.optimizer.state_dict(), 'lr': self.lr},
                   os.path.join(self.dict_paths['checkpoint'], "training_stuff_" + str(self.iteration_current) + ".pkl"))

        torch.save(self.layer_s2.state_dict(),
                   os.path.join(self.dict_paths['checkpoint'], "s2_layer_" + str(self.iteration_current) + ".pkl"))

        torch.save(self.layer_lrf.state_dict(),
                   os.path.join(self.dict_paths['checkpoint'], "lrf_layer_" + str(self.iteration_current) + ".pkl"))

    def do_step_fwd(self, data):
        signal, pts_input, signal_rot_rnd, pts_input_rot_rnd, mats_rot_rnd, name_samples = data

        # Step
        signal = signal.to(self.device)
        signal_rot_rnd = signal_rot_rnd.to(self.device)

        so3_signal = self.layer_s2(signal)

        so3_signal_rot_rnd = self.layer_s2(signal_rot_rnd)

        _, lrfs = self.layer_lrf(so3_signal)

        feature_map_trg, lrfs_rnd = self.layer_lrf(so3_signal_rot_rnd)

        mats_rot_rnd = mats_rot_rnd.to(device=self.device).float()

        lrfs_expected = torch.bmm(lrfs, mats_rot_rnd.transpose(2, 1))

        d_res_fwd = {'lrfs': lrfs,
                     'lrfs_rnd': lrfs_rnd,
                     'feature_map_trg': feature_map_trg,
                     'lrfs_expected': lrfs_expected,
                     'mats_rot_rnd': mats_rot_rnd,
                     'pts_input': pts_input,
                     'pts_input_rot_rnd': pts_input_rot_rnd,
                     'names': name_samples}

        return d_res_fwd

    def compute_loss(self, results_forward_step):

        # Loss on Local Reference Frame
        batch_loss_theta = self.criterion_theta(results_forward_step['lrfs_rnd'], results_forward_step['lrfs_expected'])
        loss_theta = torch.mean(batch_loss_theta)
        self.loss_net = loss_theta

        # Doctor Petrelli's cosine
        lrfs_src = results_forward_step['lrfs'].data.cpu().numpy()
        lrfs_rnd = results_forward_step['lrfs_rnd'].data.cpu().numpy()

        batch_lrf_repeatability = ug.lrf_repeatability(lrfs_src=np.transpose(lrfs_src, (0, 2, 1)),
                                                       lrfs_trg=np.transpose(lrfs_rnd, (0, 2, 1)),
                                                       mat_from_src_to_trg=results_forward_step['mats_rot_rnd'].data.cpu().numpy(),
                                                       th_cosine=self.th_cosine)

        d_res_batch_losses = {'theta': batch_loss_theta,
                              'lrf_rep': batch_lrf_repeatability}

        d_res_losses = {'net': self.loss_net.item(),
                        'theta': loss_theta.item(),
                        'lrf_rep': float(np.sum(batch_lrf_repeatability)) / float(self.args.size_batch)}

        return d_res_losses, d_res_batch_losses

    def compute_metric_validation(self, res_fwd_validation):
        return self.compute_loss(res_fwd_validation)

    def update_meters_train(self, res_losses, batch_losses):
        dic_avg_losses = {}

        for name, meters in self.dict_meter_loss_train.items():

            self.dict_meter_loss_train[name].add(res_losses[name])
            dic_avg_losses[name] = self.dict_meter_loss_train[name].value()[0]

        self.histogram_theta_train.append(batch_losses['theta'].data.cpu().numpy())

        return dic_avg_losses

    def update_meters_validation(self, res_losses, batch_losses):
        dic_avg_losses = {}

        for name, meters in self.dict_meter_loss_val.items():

            self.dict_meter_loss_val[name].add(res_losses[name])
            dic_avg_losses[name] = self.dict_meter_loss_val[name].value()[0]

        self.histogram_theta_validation.append(batch_losses['theta'].data.cpu().numpy())
        return dic_avg_losses

    def reset_meters_train(self):
        self.histogram_theta_train.clear()
        super().reset_meters_train()

    def reset_meters_validation(self):
        self.histogram_theta_validation.clear()
        super().reset_meters_validation()

    def __update_loss_in_visdom(self, mode_str, dict_curves):

        for title, curve in dict_curves.items():
            name_win = 'loss ' + mode_str + ' ' + title

            self.visualizer.line(X=np.arange(len(curve)),
                                 Y=np.array(curve),
                                 win=name_win.upper(),
                                 opts=dict(title=name_win, legend=[name_win], markersize=2))

    def __update_theta_borisov_histogram(self, mode_str, values):
        name_win = 'Theta ' + mode_str + ' distribution'
        self.visualizer.histogram(X=values,
                                  win=name_win,
                                  opts=dict(title=name_win, numbins=48))

    @staticmethod
    def __save_point_clouds(path, name, cloud):

        # Save pointcloud
        path_file = os.path.join(path, name + ".pcd")
        uio.save_pcd(path_file, cloud.numpy())

        print(colored('Cloud ' + name + ' saved at %s' % path, 'white', 'on_blue'))

    def update_visualizer_train(self, res_fwd, dict_avg_train_losses, dict_batch_train_losses):

        self.__update_loss_in_visdom(mode_str='Train', dict_curves=self.dict_curves_loss_train)
        self.__update_theta_borisov_histogram(mode_str='Train', values=np.asarray(self.histogram_theta_train).reshape(-1))


        idx = np.random.randint(0, dict_batch_train_losses['theta'].size(0))

        # Save pointclouds
        name_sample = res_fwd['names'][idx]
        theta_val = dict_batch_train_losses['theta'][idx]

        name_pts_input_cloud = 'pin_{:.4}_{}_'.format(theta_val, name_sample)
        cloud_in = res_fwd['pts_input'][idx].data.cpu()

        self.__save_point_clouds(path=self.dict_paths['pcd_train'], name=name_pts_input_cloud, cloud=cloud_in)

        name_pts_input_cloud_rnd = 'pin_rnd_{:.4}_{}_'.format(theta_val, name_sample)
        cloud_rnd = res_fwd['pts_input_rot_rnd'][idx].data.cpu()

        self.__save_point_clouds(path=self.dict_paths['pcd_train'], name=name_pts_input_cloud_rnd, cloud=cloud_rnd)


        lrfs_in = res_fwd['lrfs'][idx].data.cpu()
        lrfs_rnd = res_fwd['lrfs_rnd'][idx].data.cpu()
        mats_rnd = res_fwd['mats_rot_rnd'][idx].data.cpu()

        # Save lrfs
        out_file = os.path.join(self.dict_paths['npz_lrfs'], name_sample + "_{:.4}.npz".format(theta_val))
        np.savez(out_file, pts_in=cloud_in, pts_rnd=cloud_rnd, lrfs_in=lrfs_in, lrfs_rnd=lrfs_rnd, mats_rnd=mats_rnd)

        # Save visdom environment
        self.save_visualizer_env()

    def update_visualizer_validation(self, dict_curve_validation_losses, dict_curve_train_per_validation_losses):
        super().update_visualizer_validation(dict_curve_validation_losses, dict_curve_train_per_validation_losses)
        self.__update_theta_borisov_histogram('Validation', values=np.asarray(self.histogram_theta_validation).reshape(-1))

    def update_visualizer_validation_per_iteration(self, res_fwd, dict_batch_validation_losses):
        pass
        #self.__update_pointclouds_in_visdom(mode_str='Validation', res_fwd=res_fwd, dict_losses_batch=dict_batch_validation_losses)


