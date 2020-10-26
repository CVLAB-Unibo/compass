from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataloader.dataset_pointnet import ModelNetDataset
from models.network_trainer_pointnet import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

from features import lrf_compass_object as compass
from utils import torch as utor
from utils import geometry as ug


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', type=int, default=0, help="use feature transform")
parser.add_argument('--debug', type=int, default=0, help="visualize clouds")
parser.add_argument('--arbitrary_rotations', type=int, default=0, help="use random rotations on test")
parser.add_argument('--rotate_axis', type=str, default='all', help="Axis rotation when using augmentation on training/test. (possible values: 'x', 'y', 'z' and 'all')")
parser.add_argument("--path_ckp_layer_s2", type=str, required=True, help="LRF name.")
parser.add_argument("--path_ckp_layer_lrf", type=str, required=True, help="LRF name.")
parser.add_argument("--path_pointnet", type=str, default='', help="Path from pre trained weights.")
parser.add_argument("--optimized_sv", type=int, default=1, help="LRF name.")
parser.add_argument("--file_list_train", type=str, required=True, help="CSV file with folder to use in train.")
parser.add_argument("--file_list_test", type=str, required=True, help="CSV file with folder to use in test.")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

lrf_bandwidths = [24, 24, 24, 24, 24]
lrf_features = [4, 40, 20, 10, 1]
radius_support = 1.0
num_points = 2048
use_equatorial_grid = 0
softmax_temp = 1.0
device=utor.get_gpu_device(0)

lrf_estimator = compass.CompassEstimator(device=device,
                                         path_s2=opt.path_ckp_layer_s2,
                                         path_lrf=opt.path_ckp_layer_lrf,
                                         num_points=num_points,
                                         lrf_bandwidths=lrf_bandwidths,
                                         lrf_features=lrf_features,
                                         use_equatorial_grid=use_equatorial_grid,
                                         softmax_temp=softmax_temp,
                                         radius_support=radius_support,
                                         size_batch=1,
                                         num_workers=8)

if opt.dataset_type == 'modelnet40':

    if opt.path_pointnet == '':
        dataset = ModelNetDataset(root=opt.dataset,
                                  dataset=opt.dataset_type,
                                  npoints=opt.num_points,
                                  lrf=lrf_estimator,
                                  data_augmentation=False,
                                  split='trainval',
                                  file_list_folders = opt.file_list_train)

    test_dataset = ModelNetDataset(root=opt.dataset,
                                   dataset=opt.dataset_type,
                                   split='test',
                                   npoints=opt.num_points,
                                   lrf=lrf_estimator,
                                   data_augmentation=False,
                                   file_list_folders=opt.file_list_test)
else:
    exit('wrong dataset type')

if opt.path_pointnet == '':
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=int(opt.workers))
    print("Dataset train: {}".format(len(dataset)))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print("Dataset test: {}".format(len(test_dataset)))
num_classes = 40
print('classes', num_classes)

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

if opt.path_pointnet == '':

    num_batch = len(dataset) / opt.batchSize
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    for epoch in range(opt.nepoch):
        optimizer.step()
        scheduler.step()
        for i, data in enumerate(dataloader, 0):
            points, target = data

            if opt.arbitrary_rotations:
                points = ug.rotate_points(points, opt.rotate_axis)

            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 50 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data

                if opt.arbitrary_rotations:
                    points = ug.rotate_points(points, opt.rotate_axis)

                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
else:
    utor.load_models_from_ckp(opt.path_pointnet, classifier)


total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data

    if opt.arbitrary_rotations:
        points = ug.rotate_points(points, opt.rotate_axis)

    for i in range(points.shape[0]):
        lrf_estimator.radius_support = ug.get_max_radius(np.asarray(points[i]))
        lrf = lrf_estimator(np.asarray(points[i]))
        points[i] = points[i] @ lrf[0].T


    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))