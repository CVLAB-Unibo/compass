import glob
import os


def make_dir(path_dir):
    """
    Create a directory if not exists
    :param path_dir: path to directory
    """

    if not os.path.exists(path_dir) and not os.path.isdir(path_dir):
        try:
            os.makedirs(path_dir)
        except OSError as exception:
            print("Error while make dirs: ".format(exception))