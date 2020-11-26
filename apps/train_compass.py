import argparse
import yaml
from models import network_trainer_compass as ntc


def parse_commandline():
    parser = argparse.ArgumentParser(description="Compass Network training arguments.")

    """Training"""
    parser.add_argument("--config_file", type=str, required=True, help="Configuration file for training.")
    parser.add_argument("--name_train", type=str, required=True, help="Name of the training.")
    parser.add_argument("--path_log", type=str, required=True, help="Path to log directory.")

    # Dataset
    parser.add_argument("--path_ds", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--name_file_folder_train", type=str, required=True, help="CSV file specifying the objects/scenes of the dataset to use in train.")
    parser.add_argument("--name_file_folder_validation", type=str, required=True, help="CSV file specifying the objects/scenes of the dataset to use in validation.")
    
    # To set only for Test-Time Adaptation, or to continue a previous training
    parser.add_argument("--path_ckp_ts", type=str, default=None, help="Path to pretrained training stuff.")
    parser.add_argument("--path_ckp_s2_layer", type=str, default=None, help="Path to pretrained S2 layer.")
    parser.add_argument("--path_ckp_lrf_layer", type=str, default=None, help="Path to pretrained lrf layer.")
    

    # Misc
    parser.add_argument("--port_vis", type=int, default=8888, help="Port for visdom sever.")

    parser.add_argument("--use_gpu", type=int, default=1, help="Default 1, set it to 0 to use the CPU only. WARNING: this slows down the computation, use only for testing purposes or if you don't have a CUDA capable GPU.")
    parser.add_argument("--id_gpu", type=int, default=0, help="Id of the GPU to use to train.")
    parser.add_argument("--num_workers", type=int, default=6, help="Num of workers for data loader (#CPU Cores - 1).")
    
    parser.add_argument("--size_batch", type=int, default=8, help="Size of batch. Affects training and also GPU VRAM occupation.")
    

    return parser.parse_args()


def main(args):

    network_trainer = ntc.CompassNetworkTrainer(args)

    network_trainer.prepare()

    network_trainer.start_train()

    network_trainer.shutdown()


if __name__ == "__main__":

    arguments = parse_commandline()

    with open(arguments.config_file) as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)
        print(params)
        for key, value in params.items():
            setattr(arguments, key, value)

    main(arguments)