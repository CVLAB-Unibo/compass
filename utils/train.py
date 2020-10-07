import logging
import os
import numpy as np
import sys

from termcolor import colored

from models.loss import ChamferLoss


def get_console_string(date, iteration, epoch, index_batch, len_loader, learning_rate, loss_net, time_fwd):

    string_losses = ' '.join('{}: {:.4} '.format(key, val) for key, val in loss_net.items())

    console_string = "[{}]-[{}]-[{}:{}/{}] " \
                     "Lr: {:.4} " \
                     "{}" \
                     "Time: fw={:.2}".format(date,
                                             iteration,
                                             epoch,
                                             index_batch,
                                             len_loader,
                                             learning_rate,
                                             string_losses,
                                             time_fwd)

    return console_string

def setup_logger(name_logger, path_log_file, name_log_file, level=logging.DEBUG):

    """
    Setup logger for train
    :param name_logger: name for the logger
    :param path_log_file: path to log file
    :param name_log_file: name of the log file
    :param level: level for logger default DEBUG
    :return:
    """

    logger = logging.getLogger(name_logger)
    logger.setLevel(level)
    logger.handlers = []

    hand_stream = logging.StreamHandler()
    logger.addHandler(hand_stream)

    hand_file = logging.FileHandler(os.path.join(path_log_file, name_log_file))
    logger.addHandler(hand_file)

    return logger


def save_pcd(mode,
             iteration,
             cloud_in,
             cloud_in_rnd,
             name_sample,
             path_cloud):

    # Save pcd in and out
    name = name_sample + "_it_" + str(iteration)
    path_file = os.path.join(path_cloud, name + "_input" + ".pcd")
    utils_io.save_pcd(path_file, cloud_in.numpy())

    path_file = os.path.join(path_cloud, name + "_random" + ".pcd")
    utils_io.save_pcd(path_file, cloud_in_rnd.numpy())

    print(colored('Cloud ' + mode + ' saved at %s' % path_cloud, 'white', 'on_blue'))
    