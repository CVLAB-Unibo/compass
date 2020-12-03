import open3d as o3d
import copy
import csv
import datetime as dt
import logging
import os
import time
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import math
import numpy as np

from termcolor import colored

from dataloader import spherical_voxel as sv

from features import det_uniform_sampling as kpus
from features import lrf_compass as compassnet

from models import loss as l
from models import network_s2_layer as nsl
from models import network_lrf_layer as nll

from utils import io as uio
from utils import file_system as ufs
from utils import geometry as ug
from utils import visualization as uvz
from utils import torch as utor
from utils import progress_bar


class LocalReferenceFrameBenchmark():

    def __init__(self, arguments):

        self.args = arguments
        self.logger = logging.getLogger("LRFBenchmarkLogger")
        self.duration = 0.0
        self.mean_chamfer = []
        self.mean_theta = []
        self.mean_lrf_repeatability = []
        self.file_results = None
        self.writer_results = None
        self.detector = None
        self.lrf_estimator = None

        self.name_file_current_cloud = None

        # Added for PyFlare on Meshes
        self.faces_src = None
        self.faces_trg = None

        # PyFlare works also on meshes, for the other method support is only for 3DMatch
        self.name_file_mat_gt = self.args.name_file_gt
        self.name_file_overlap = self.args.name_file_overlap
        self.name_dataset = self.args.name_data_set
        self.overlap_threshold = self.args.overlap_threshold
        
        self.metric_points = None
        self.metric_rotations = None
        self.metric_lrf_repeatability = None

    def init_logger(self):

        pth_file_log = os.path.join(self.args.path_results, "log.txt")

        self.logger.setLevel(level=logging.DEBUG)
        self.logger.handlers = [logging.StreamHandler(), logging.FileHandler(pth_file_log)]

        return True

    def init_file_results(self):

        pth_file_results = os.path.join(self.args.path_results, "results.csv")

        self.file_results = open(pth_file_results, 'w')

        self.writer_results = csv.writer(self.file_results, delimiter=';')
        self.writer_results.writerow(['Fragment', 'Cloud_Src', 'Cloud_Trg', 'I_Kp_Src', 'I_Kp_Trg', 'Chamfer',
                                      'Theta_Rot', 'LRF_Repeatability%'])

        return True

    def init_file_pairs(self):
        pth_file_results = os.path.join(self.args.path_results, "pairs.csv")

        self.file_results = open(pth_file_results, 'w')

        self.writer_pairs = csv.writer(self.file_results, delimiter=';')
        self.writer_pairs.writerow(['Fragment', 'Cloud_Src', 'Cloud_Trg', 'Num_kp', 'Overlap', 'Angle',
                                      'Theta_Rot', 'LRF_Repeatability%'])

        return True

    def init_lrf_estimator(self):
        if (self.args.use_gpu >= 1):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)

            self.device = utor.get_gpu_device(0)
            torch.backends.cudnn.benchmark = True
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
            print(colored("WARNING: Running on CPU only", 'white', 'on_red'))

            self.device = utor.get_cpu_device()


        self.lrf_estimator = compassnet.CompassEstimator(device=self.device,
                                                        path_s2=self.args.path_ckp_layer_s2,
                                                        path_lrf=self.args.path_ckp_layer_lrf,
                                                        num_points=self.args.size_point_cloud,
                                                        lrf_bandwidths=self.args.lrf_bandwidths,
                                                        lrf_features=self.args.lrf_features,
                                                        use_equatorial_grid=0,
                                                        softmax_temp=1.0,
                                                        radius_support=self.args.radius_descriptor,
                                                        size_batch=self.args.size_batch,
                                                        num_workers=self.args.num_workers )
        self.logger.info(self.lrf_estimator)

        return True

    def init_keypoints_detector(self):

        self.detector = kpus.UniformSamplingKeypointDetector(radius=self.args.radius_detector)
        self.logger.info(self.detector)

        return True

    def init_metrics(self):
        self.metric_points = l.ChamferLoss()
        self.metric_rotations = l.ThetaBorisovLoss(device=torch.device('cpu'))
        self.metric_lrf_repeatability = ug.lrf_repeatability

        return True

    def estimate_lrfs(self, cloud, indices, faces):
        return self.lrf_estimator(cloud, indices, return_features_map=not bool(self.args.is_batch))

    def get_pairs_and_gt_mat(self):
        dict_ds_loader = {"3DMatch": uio.get_pair_and_matrix_3dmatch,
                          "StanfordViews": uio.get_pairs_and_matrix_stanford}

        return dict_ds_loader[self.name_dataset]((os.path.join(self.args.path_ds, self.name_file_mat_gt)))

    def get_overlapping_areas(self):
        dict_overlap_loader = {"3DMatch": uio.get_overlapping_areas_3dmatch,
                                "StanfordViews": uio.get_overlapping_areas_stanford}
        
        if (self.name_file_overlap != None):
            return dict_overlap_loader[self.name_dataset]((os.path.join(self.args.path_ds, self.name_file_overlap)))
        else:
            return {}

    def read_source_and_target_cloud(self, pair, cloud_file_name):
        if self.name_dataset == "3DMatch":
            idx_src = int(pair.split("-")[0])
            idx_trg = int(pair.split("-")[1])

            # Load cloud
            name_cloud_src = "{}_{}.{}".format(cloud_file_name, idx_src, self.args.ext_cloud)
            path_cloud_src = os.path.join(self.args.path_ds, name_cloud_src)
            cloud_src = o3d.io.read_point_cloud(path_cloud_src)

            name_cloud_trg = "{}_{}.{}".format(cloud_file_name, idx_trg, self.args.ext_cloud)
            path_cloud_trg = os.path.join(self.args.path_ds, name_cloud_trg)
            cloud_trg = o3d.io.read_point_cloud(path_cloud_trg)

            if (self.args.leaf > 0):
                cloud_src = cloud_src.voxel_down_sample(self.args.leaf)
                cloud_trg = cloud_trg.voxel_down_sample(self.args.leaf)

        # Open3D can not subsample 3DMesh
        if self.name_dataset == "StanfordViews":
            name_cloud_src = "{}.{}".format(pair.split("-")[0], self.args.ext_cloud)
            name_cloud_trg = "{}.{}".format(pair.split("-")[1], self.args.ext_cloud)

            path_cloud_src = os.path.join(self.args.path_ds, name_cloud_src)
            mesh_src = o3d.io.read_triangle_mesh(path_cloud_src)
            self.faces_src = mesh_src.triangles
            cloud_src = o3d.geometry.PointCloud()
            cloud_src.points = o3d.utility.Vector3dVector(mesh_src.vertices)

            path_cloud_trg = os.path.join(self.args.path_ds, name_cloud_trg)
            mesh_trg = o3d.io.read_triangle_mesh(path_cloud_trg)
            self.faces_trg = mesh_trg.triangles
            cloud_trg = o3d.geometry.PointCloud()
            cloud_trg.points = o3d.utility.Vector3dVector(mesh_trg.vertices)

        return cloud_src, cloud_trg, name_cloud_src, name_cloud_trg

    def store_sample(self, sample):

        val_list = []

        for value in sample.values():
            val_list.append(value)

        self.writer_results.writerow(val_list)

    def store_pair(self, pair):

        val_list = []

        for value in pair.values():
            val_list.append(value)

        self.writer_pairs.writerow(val_list)

    def visualize_sample(self, sample):

        cloud_src = o3d.geometry.PointCloud()
        cloud_src.points = o3d.utility.Vector3dVector(sample['src'])
        cloud_src.paint_uniform_color([1, 0, 0])

        cloud_trg = o3d.geometry.PointCloud()
        cloud_trg.points = o3d.utility.Vector3dVector(sample['trg'])
        cloud_trg.paint_uniform_color([0, 1, 0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0 * self.args.min_distance, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([cloud_src, cloud_trg, frame])

    def print_sample(self, sample):
        print("Kp src-trg: {}-{} - Chamfer: {} - Theta: {} - Is Rep: {}".format(sample['i_kp_src'], sample['i_kp_trg'], sample['chamfer'], sample['theta'], sample['is_rep']))

    def prepare(self):

        # Create out folder
        ufs.make_dir(self.args.path_results)

        if not self.init_logger():
            raise Exception('init_logger() failed.')

        if not self.init_file_results():
            raise Exception('init_file_results() failed.')

        if not self.init_file_pairs():
            raise Exception('init_file_pairs() failed.')

        if not self.init_keypoints_detector():
            raise Exception('init_keypoints_detector() failed.')

        if not self.init_lrf_estimator():
            raise Exception('init_lrf_estimator() failed.')

        if not self.init_metrics():
            raise Exception('init_metrics() failed.')

    def start(self):
        # Save arguments
        self.logger.info("%s", repr(self.args))

        self.logger.info("Benchmark start at [{}]".format(dt.datetime.now()))

        dict_pairs_mat = self.get_pairs_and_gt_mat()

        self.logger.info("Loaded: {} pairs from: {}".format(len(dict_pairs_mat), self.args.path_ds))

        dict_overlap = self.get_overlapping_areas()

        # If the dictionary is None or empty, compute on all the pairs
        # Otherwise, compute only on pairs for which you know the overlap, and the overlap
        #   exceeds the set threshold
        if (dict_overlap != None and len(dict_overlap) > 0):
            self.logger.info("Loaded: {} overlapping areas from: {}".format(len(dict_overlap), self.args.path_ds))

            dict_pairs_mat = {pair:mat for (pair, mat) in dict_pairs_mat.items() if pair in dict_overlap and dict_overlap[pair] > self.overlap_threshold}

            self.logger.info("{} pairs with overlapping area > {}".format(len(dict_pairs_mat), self.overlap_threshold))

        # Find name of cloud file
        cloud_file_name = ""
        for file in os.listdir(self.args.path_ds):
            if file.endswith("." + self.args.ext_cloud):
                cloud_file_name = str(file)
                break

        list_expr = re.findall(r"(.*)_[^_]*\." + self.args.ext_cloud + "$", cloud_file_name)
        cloud_file_name = ""

        if len(list_expr):
            cloud_file_name = list_expr[0]

        for idx_pair, (pair, mat_gt_trg_to_src) in enumerate(dict_pairs_mat.items()):
            name_fragment = self.args.path_ds.split(os.sep)[-2]


            pair_lrf_repeatabilities = []
            pair_theta_rot = []
            # Ground truth Theta angle between src and trg
            pair_angle_gt = self.metric_rotations(torch.tensor(mat_gt_trg_to_src[:3, :3]).view(1, 3, 3), torch.eye(3, device='cpu').reshape((1,3,3)))[0]

            cloud_src, cloud_trg, name_cloud_src, name_cloud_trg = self.read_source_and_target_cloud(pair,
                                                                                        cloud_file_name)

            overlap_string = str(dict_overlap[pair]) if pair in dict_overlap else "{:.4}".format(ug.get_overlapping_area(cloud_a=cloud_src,
                                                       cloud_b=cloud_trg,
                                                       mat_pose_a=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
                                                       mat_pose_b=mat_gt_trg_to_src,
                                                       min_euclidean_distance_to_consider_kp_in_fragment=self.args.min_distance))
            
            self.logger.info("Pair:{}   Overlap: {}    Angle: {:.4} ({:.3} deg)".format(pair, overlap_string, pair_angle_gt, pair_angle_gt / np.pi * 180))


            cloud_kp_src, indices_kp_src = self.detector(cloud_src)

            # Perfect detector
            cloud_kp_src_in_trg = copy.deepcopy(cloud_kp_src)
            cloud_kp_src_in_trg.transform(np.linalg.inv(mat_gt_trg_to_src))

            cloud_kp_trg, indices_kp_trg, indices_src_overlap = ug.compute_nearest_search_intersection(cloud_kp_src_in_trg,
                                                                                                       cloud_trg,
                                                                                                       self.args.min_distance)
            if indices_src_overlap.shape[0] == 0:
                print("No overlapping keypoints in the current pair")
                continue

            indices_kp_src = indices_kp_src[indices_src_overlap].reshape(indices_src_overlap.shape[0])
            cloud_kp_src = o3d.geometry.PointCloud()
            cloud_kp_src.points = o3d.utility.Vector3dVector(np.asarray(cloud_src.points)[indices_kp_src])

            self.logger.info("Keypoints overlap: src {} - trg {}".format(indices_kp_src.shape[0], indices_kp_trg.shape[0]))

            if not bool(self.args.is_batch):
                viz_src = uvz.viz_keypoints(cloud_src, indices_kp_src, self.args.radius_descriptor,
                                            color_cloud=[1, 0, 0], color_keypoints=[0, 0, 0])
                viz_trg = uvz.viz_keypoints(cloud_trg, indices_kp_trg, self.args.radius_descriptor,
                                            color_cloud=[0, 1, 0], color_keypoints=[0, 0, 0])
                o3d.visualization.draw_geometries(viz_src + viz_trg)

            time_start = time.time()

            # Batch split used only in visualization mode to improve responsiveness,
            # otherwise single batch with all the keypoints to improve performance
            if not bool(self.args.is_batch):
                num_splits = len(indices_kp_src)
            else:
                num_splits = 1
            indices_kp_src_in_batch = np.array_split(indices_kp_src, num_splits)
            indices_kp_trg_in_batch = np.array_split(indices_kp_trg, num_splits)

            for batch_i in range(len(indices_kp_src_in_batch)):

                self.name_file_current_cloud = name_cloud_src
                if bool(self.args.is_batch):
                    print("Estimating LRFs from Source cloud = {}".format(self.name_file_current_cloud))
                lrfs_src, pts_src, lrf_features_map_src = self.estimate_lrfs(cloud_src,
                                                                             indices_kp_src_in_batch[batch_i],
                                                                             self.faces_src)

                self.name_file_current_cloud = name_cloud_trg
                if bool(self.args.is_batch):
                    print("Estimating LRFs from Target cloud = {}".format(self.name_file_current_cloud))
                lrfs_trg, pts_trg, lrf_features_map_trg = self.estimate_lrfs(cloud_trg,
                                                                             indices_kp_trg_in_batch[batch_i],
                                                                             self.faces_trg)

                with torch.no_grad():
                    # Compute Chamfer
                    lrfs_src = torch.tensor(lrfs_src, dtype=torch.float, device=self.device).view(-1, 3, 3)
                    pts_src = torch.tensor(pts_src, dtype=torch.float, device=self.device).view(-1, self.args.size_point_cloud, 3)
                    pts_src_in_lrf = utor.rotate_batch_cloud(pts_src, lrfs_src)

                    lrfs_trg = torch.tensor(lrfs_trg, dtype=torch.float, device=self.device).view(-1, 3, 3)
                    pts_trg_tensor = torch.tensor(pts_trg, dtype=torch.float, device=self.device).view(-1, self.args.size_point_cloud, 3)
                    pts_trg_in_lrf = utor.rotate_batch_cloud(pts_trg_tensor, lrfs_trg)

                    chamfer_batch = 10
                    iterations = int(np.ceil(pts_src_in_lrf.shape[0] / chamfer_batch))
                    dist_src_trg = torch.empty((pts_src_in_lrf.shape[0], pts_src_in_lrf.shape[1]), dtype=torch.float)
                    dist_trg_src = torch.empty((pts_src_in_lrf.shape[0], pts_src_in_lrf.shape[1]), dtype=torch.float)
                    
                    for cham_i in range(iterations):
                        upper_limit = min(pts_src_in_lrf.shape[0], (cham_i+1)*chamfer_batch)
                        dist_src_trg[cham_i*chamfer_batch:upper_limit,:], dist_trg_src[cham_i*chamfer_batch:upper_limit,:] = self.metric_points(pts_src_in_lrf[cham_i*chamfer_batch:upper_limit,:,:], pts_trg_in_lrf[cham_i*chamfer_batch:upper_limit,:,:])

                    distances_chamfer = torch.mean(dist_src_trg, dim=1) + torch.mean(dist_trg_src, dim=1)

                    # Compute Theta
                    mat_gt_src_trg_tensor = torch.tensor(mat_gt_trg_to_src[:3, :3].T).view(1, 3, 3)
                    batch_gt_tensor = mat_gt_src_trg_tensor.repeat(lrfs_src.shape[0], 1, 1)

                    lrfs_expected = torch.bmm(lrfs_src.to('cpu'), batch_gt_tensor.transpose(2, 1))
                    distances_rotation = self.metric_rotations(lrfs_trg.to('cpu'), lrfs_expected)

                    # Th cosines
                    lrfs_src_as_cols = np.transpose(lrfs_src.data.cpu().numpy(), (0, 2, 1))
                    lrfs_trg_as_cols = np.transpose(lrfs_trg.data.cpu().numpy(), (0, 2, 1))
                    mat_gt_src_to_trg = mat_gt_trg_to_src[:3, :3].T

                    lrf_repeatability = self.metric_lrf_repeatability(lrfs_src=lrfs_src_as_cols,
                                                                      lrfs_trg=lrfs_trg_as_cols,
                                                                      mat_from_src_to_trg=mat_gt_src_to_trg,
                                                                      th_cosine=self.args.th_cosine)

                # Save Results
                for i in range(indices_kp_src_in_batch[batch_i].shape[0]):
                    row = {'fragment': name_fragment,
                           'c_src': name_cloud_src,
                           'c_trg': name_cloud_trg,
                           'i_kp_src': "{}".format(indices_kp_src_in_batch[batch_i][i]),
                           'i_kp_trg': "{}".format(indices_kp_trg_in_batch[batch_i][i]),
                           'chamfer': "{:.6}".format(distances_chamfer[i]),
                           'theta': "{:.6}".format(distances_rotation[i]),
                           'is_rep': "{0}".format(lrf_repeatability[i]) }

                    pair_lrf_repeatabilities.append(lrf_repeatability[i])
                    pair_theta_rot.append(distances_rotation[i])

                    # Update means
                    self.mean_chamfer.append(distances_chamfer[i])
                    self.mean_theta.append(distances_rotation[i])
                    self.mean_lrf_repeatability.append(lrf_repeatability[i])

                    self.store_sample(row)
                    
                    if not self.args.is_batch:
                        sample_viz = {'i_kp_src': "{}".format(indices_kp_src_in_batch[batch_i][i]),
                                        'i_kp_trg': "{}".format(indices_kp_trg_in_batch[batch_i][i]),
                                        'src': pts_src_in_lrf[i].data.cpu().numpy(),
                                        'trg': pts_trg_in_lrf[i].data.cpu().numpy(),
                                        'chamfer': "{:.6}".format(distances_chamfer[i]),
                                        'theta': "{:.6}".format(distances_rotation[i]),
                                        'is_rep': "{0}".format(lrf_repeatability[i]) }

                        # if lrf_repeatability[i] == 0:
                        #     continue

                        self.print_sample(sample_viz)
                        self.visualize_sample(sample_viz)
                        

                torch.cuda.empty_cache()



            # Save pair-wise information (angle and repeatability)
            pair_lrf_rep_mean = np.sum(pair_lrf_repeatabilities) / len(pair_lrf_repeatabilities)
            pair_theta_rot_mean = np.mean(pair_theta_rot)
            pair_info = {'fragment': name_fragment,
                           'c_src': name_cloud_src,
                           'c_trg': name_cloud_trg,
                           'num_kp': indices_kp_src.shape[0],
                           'overlap': overlap_string,
                           'angle': "{:.6}".format(pair_angle_gt),
                           'theta': pair_theta_rot_mean,
                           'lrf_rep': pair_lrf_rep_mean
            }
            self.store_pair(pair_info)

            time_end = time.time() - time_start
            self.duration += time_end

            current_lrf_rep = np.sum(self.mean_lrf_repeatability) / len(self.mean_lrf_repeatability)
            self.logger.info("[{}/{}] Pair:{} - Time {:.4} sec - Cumulative LRF Rep: {:.4} - LRF Rep of this pair only: {:.4}".format(idx_pair+1, len(dict_pairs_mat), pair, time_end, current_lrf_rep, pair_lrf_rep_mean))

        self.logger.info("Benchmark end at [{}]".format(dt.datetime.now()))

    def print_results(self):

        self.logger.info("=============================")
        self.logger.info("Benchmark is over")
        self.logger.info("Duration: {} seconds".format(self.duration))
        self.logger.info("Average Chamfer distance: {:.6}".format(np.mean(self.mean_chamfer)))
        self.logger.info("Average Theta Borisov: {:.6}".format(np.mean(self.mean_theta)))
        self.logger.info("Average LRF repeatability: {:.6}".format(np.sum(self.mean_lrf_repeatability) / len(self.mean_lrf_repeatability)))
        self.logger.info("=============================")

    def store_results(self):
        # Store histogram
        plt.hist(self.mean_theta, bins=50)
        plt.savefig(os.path.join(self.args.path_results, "histogram.png"), bbox_inches='tight')
        plt.clf()
        
        # Store meters
        np.save(os.path.join(self.args.path_results, "chamfer"), self.mean_chamfer, allow_pickle=False)
        np.save(os.path.join(self.args.path_results, "theta"), self.mean_theta, allow_pickle=False)
        np.save(os.path.join(self.args.path_results, "repet"), self.mean_lrf_repeatability, allow_pickle=False)
        
        self.logger.info("Results saved in {}".format(self.args.path_results))
        self.file_results.close()

    def shutdown(self):
        pass
