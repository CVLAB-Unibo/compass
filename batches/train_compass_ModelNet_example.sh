#!/bin/bash
## INSTRUCTIONS
# This batch file is used for train compass in the ModelNet dataset
#
# First download ModelNet dataset from: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
# For more information about this dataset please refer to: https://github.com/charlesq34/pointnet
#
# Create files named train.csv and validation.csv
# We adopt files ply_data_train[0:3].h5 for train and ply_data_train4.h5 for validation
#
# If you find this work useful please cite

date=`date +%Y-%m-%d`
nm_train="Compass_ModelNet"
dir_ds="/media/mmarcon/data/DATASETS/modelnet40_ply_hdf5_2048/" #path to the dataset
date_time=`date +%Y-%m-%d_%H-%M`
dir_log="/media/mmarcon/data/Train_SOUND/ModelNet/$date_time-$nm_train" #path to save training files
ep_max=100
file_train="$dir_ds/train.csv"
file_validation="$dir_ds/validation.csv"
ext="h5"
rad_desc=0.30
num_work=12
num_pts=2048
sz_btch=8
stp_save=200
stp_viz=50
name_data_set="ModelNet"

export PYTHONPATH="/home/mmarcon/workspace_python/compass" #put your workspace's path

python3 ../apps/train_compass.py  --lrf_bandwidths 24 24 24 24 24 --lrf_features 4 40 20 10 1 \
	                              --ext $ext --epochs_max $ep_max \
	                              --path_ds $dir_ds --path_log $dir_log --num_workers $num_work \
	                              --size_batch $sz_btch --size_bandwidth 24 --size_channels 4 --size_pcd $num_pts \
	                              --step_per_save $stp_save --step_per_viz $stp_viz --port_vis 8888 --name_file_folder_train $file_train \
	                              --name_file_folder_validation $file_validation \
	                              --radius_descriptor $rad_desc --name_train $nm_train \
	                              --leaf_sub_sampling 0 --name_data_set $name_data_set
