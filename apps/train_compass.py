import argparse
from models import network_trainer_compass as ntc


def parse_commandline():
    parser = argparse.ArgumentParser(description="SONIC-LRF Network train arguments.")

    """Training"""
    parser.add_argument("--size_channels", type=int, required=True, help="Size of input channels.")
    parser.add_argument("--size_bandwidth", type=int, required=True, help="Size of the input bandwidth.")
    parser.add_argument('--lrf_bandwidths', type=int, nargs='+', required='True', help='List of input and ouput bandwidths for the network')
    parser.add_argument('--lrf_features', type=int, nargs='+', required='True', help='List of features size for the network')
    parser.add_argument("--augmentation", type=int, default=1, help="Data augmentation with multiple random rotations.")
    parser.add_argument("--ext", choices={"h5", "pcd", "ply"}, default="ply", help="Dataset files extension.")
    parser.add_argument("--epochs_max", type=int, required=True, help="Number of max epochs.")
    parser.add_argument("--id_gpu", help="Id of gpu to use to train", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--leaf_sub_sampling", type=float, required=True, help="Leaf for uniform sampling.")
    parser.add_argument("--leaf_keypoints", type=float, default=0.1, help="Leaf for keypoints sampling.")
    parser.add_argument("--path_ds", type=str, required=True, help="Path to dataset.")
    parser.add_argument("--path_ckp_ts", type=str, default=None, help="Path to pretrained training stuff.")
    parser.add_argument("--path_ckp_s2_layer", type=str, default=None, help="Path to pretrained S2 layer.")
    parser.add_argument("--path_ckp_lrf_layer", type=str, default=None, help="Path to pretrained decoder.")
    parser.add_argument("--path_log", type=str, required=True, help="Path to log directory.")
    parser.add_argument("--port_vis", type=int, default=8888, required=True, help="Port for visdom sever.")
    parser.add_argument("--path_models", type=str, default="../models", help="Path to models directory.")
    parser.add_argument("--radius_descriptor", type=float, help="Radius support for local geometry.")
    parser.add_argument("--name_train", type=str, required=True, help="Name of the train.")
    parser.add_argument("--name_file_folder_train", type=str, required=True, help="CSV file with folder to use in train.")
    parser.add_argument("--name_file_folder_validation", type=str, required=True, help="CSV file with folder to use in validation.")
    parser.add_argument("--num_workers", type=int, default=1, help="Num of workers for data loader.")
    parser.add_argument("--size_batch", type=int, default=32, help="Size of batch.")
    parser.add_argument("--size_pcd", type=int, required=True, help="Number point for the reconstructed point cloud.")
    parser.add_argument("--step_per_save", type=int, default=500, help="Export model and do validation every step_per_save.")
    parser.add_argument("--step_per_viz", type=int, default=15, help="Update visualizer every step_per_viz.")
    parser.add_argument("--use_gpu", type=int, default=1, help="Default 1, set it to 0 to use the CPU only. WARNING: this slows down the computation, use only for testing purposes or if you don't have a CUDA capable GPU.")

    parser.add_argument("--name_data_set", type=str, default="3DMatch", help="Name of dataset: 3DMatch, StanfordViews, ModelNet, and ShapeNet.")

    parser.add_argument("--removal_augmentation", type=int, default=0, help="Randomly delete portions of the input to increase robustness.")

    return parser.parse_args()


def main(args):

    network_trainer = ntc.CompassNetworkTrainer(args)

    network_trainer.prepare()

    network_trainer.start_train()

    network_trainer.shutdown()


if __name__ == "__main__":

    arguments = parse_commandline()
    main(arguments)