from __future__ import print_function
import torch.utils.data as data
import os
import csv
import glob
import os.path
import torch
import numpy as np
from utils import geometry as ug
from utils import io as uio

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 file_list_folders,
                 dataset="modelnet40",
                 lrf=None,
                 npoints=2500,
                 split='train',
                 data_augmentation=True,
                 process_compass=False):
        self.dataset = dataset
        self.file_list_folders = file_list_folders
        self.lrf_estimator = lrf
        self.npoints = npoints
        self.root = root
        self.split = split
        self.num_classes = 1
        self.data_augmentation = data_augmentation
        self.clouds = np.array([], dtype=np.float32).reshape(0, 2048 ,3)
        self.labels = np.array([], dtype=np.int64).reshape(0, self.num_classes)
        self.files = []
        self.process_compass = process_compass

        if self._check_exists():

            for f in self.files:
                data, labels = uio.load_h5(f)

                print ("Processing {}...".format(f))

                #Pre orient input according to compass
                if (self.process_compass):
                    for i in range(len(data)):
                        self.lrf_estimator.radius_support = ug.get_max_radius(np.asarray(data[i]))
                        lrf = self.lrf_estimator(np.asarray(data[i]))
                        data[i] = data[i] @ lrf[0].T

                        if i % 100 == 0:
                            print("[{} of {}]". format(i, len(data)))

                self.clouds = np.vstack((self.clouds, data))
                self.labels = np.vstack((self.labels, labels))
        else:
            raise RuntimeError('Dataset not found in {}.'.format(self.root))


    def _check_exists(self):

        with open(self.file_list_folders) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')

            for row in csv_reader:
                if(self.dataset == "modelnet40"):
                    illegal_name = os.path.join(self.root, str(row[0]), row[0])
                    filenames = glob.glob(os.path.join(self.root, str(row[0])))
                    filenames = [filename for filename in filenames if filename != illegal_name]
                    self.files += filenames

        return len(self.files) > 0


    def __getitem__(self, index):
        pts = self.clouds[index, :, :]
        cls = self.labels[index][0]

        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        return point_set, cls

    def __len__(self):
        return self.labels.shape[0]


